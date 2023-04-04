import os
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import yaml
import imageio

import datasets
from tqdm import tqdm
from torchvision.utils import save_image
from skimage import img_as_ubyte
import cv2

from skimage.transform import resize
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim

from modules.Facevid.keypoint_detector import KPDetector, HEEstimator
from modules.Facevid.generator_facevid import OcclusionAwareGenerator
from modules.Facevid.keypoint_transform import keypoint_transformation
from modules.FOMM.keypoint_detector import KPDetector_FOMM
from modules.FOMM.generator import OcclusionAwareGenerator_FOMM
from modules.deca.decalib.deca import DECA
from modules.diffusion.diffusion_generator import DiffusionGenerator

from frames_dataset import DatasetRepeater, FramesDataset


def set_diffusion(device, opt, img_size=256, time_steps=100):
    diffusion_model = DiffusionGenerator(img_size=img_size, timesteps=time_steps)
    if opt.load_dir != '':
        
        checkpoint = torch.load(os.path.join(opt.load_dir, 'diffusion_generator.pth'), map_location='cpu')
        diffusion_model.load_state_dict(checkpoint)
        
        print('load diffusion model successfull')
    diffusion_model = diffusion_model.to(device)
    return diffusion_model


def set_optimizer(diffusion_ddp, opt):
    optimizer = torch.optim.Adam(diffusion_ddp.parameters(), lr=2e-4, betas=(0, 0.9))

    if opt.load_dir != '':
        optimizer.load_state_dict(
            torch.load(os.path.join(opt.load_dir, 'optimizer.pth'), map_location='cpu'))
        print('load optimizer successfull')

    return optimizer


def set_face_vid(device, load_path, config):
    checkpoint = torch.load(load_path, map_location='cpu')

    keypoint_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                   **config['model_params']['common_params'])
    keypoint_detector.load_state_dict(checkpoint['kp_detector'])
    keypoint_detector.to(device)

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    he_estimator.load_state_dict(checkpoint['he_estimator'])
    he_estimator.to(device)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)

    keypoint_detector.eval()
    he_estimator.eval()
    generator.eval()

    return keypoint_detector, he_estimator, generator


def set_fomm(device, load_path, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    checkpoint = torch.load(load_path, map_location='cpu')
    kp_detector = KPDetector_FOMM(**config['model_params']['kp_detector_params'],
                                  **config['model_params']['common_params'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    kp_detector.to(device)
    
    generator = OcclusionAwareGenerator_FOMM(**config['model_params']['generator_params'],
                                             **config['model_params']['common_params'])
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)

    kp_detector.eval()
    generator.eval()

    return kp_detector, generator


@torch.no_grad()
def get_trans_img(source_img, driving_img, keypoint_detector, he_estimator, generator):
    kp_c = keypoint_detector(source_img)

    he_s = he_estimator(source_img)
    he_d = he_estimator(driving_img)

    estimate_jacobian = False
    kp_source = keypoint_transformation(kp_c, he_s, estimate_jacobian)
    kp_driving = keypoint_transformation(kp_c, he_d, estimate_jacobian)
    out = generator(source_img, kp_source=kp_source, kp_driving=kp_driving)

    return out


@torch.no_grad()
def get_trans_img_fomm(source_img, driving_img, keypoint_detector, generator):
    kp_s = keypoint_detector(source_img)
    kp_d = keypoint_detector(driving_img)
    out = generator(source_img, kp_driving=kp_d, kp_source=kp_s)

    return out


@torch.no_grad()
def diffusion_condition(s, d, condition, deca, train=True):

    _, attr_s, attr_d = deca(source_pic=s, driving_pic=d)
    attr = torch.cat((attr_s['exp'], attr_s['pose']), dim=1) - torch.cat((attr_d['exp'], attr_d['pose']), dim=1)

    low_feat = [F.interpolate(condition['prediction'], size=(32, 32)), 
                F.interpolate(condition['prediction'], size=(64, 64)), 
                F.interpolate(condition['prediction'], size=(128, 128))]
    g = F.interpolate(condition['prediction'], size=(32, 32))
    if train:
        d = F.interpolate(d, size=(32, 32))
    else:
        d = F.interpolate(condition['prediction'], size=(32, 32))
    g_dict = {'generated': F.interpolate(g, size=(256, 256)), 'driving': F.interpolate(d, size=(256, 256))}

    condition_img = {'generated': g_dict, 'warping': low_feat, 'attribute': attr}
    return condition_img


def save_255_image(img_tensor, path):
    x = 255 * (np.transpose(img_tensor.data.cpu().numpy(), [0, 2, 3, 1])[0])
    x = np.clip(x, 0, 255)
    return (x).astype(np.uint8)


def training_process(rank, world_size, opt, device):

    diffusion_model = set_diffusion(device, opt)
    diffusion_ddp = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)

    diffusion_model = diffusion_ddp.module
    optimizer = set_optimizer(diffusion_ddp, opt)

    config_path = 'config/vox-256.yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    load_path = 'modules/Facevid/ckpt/Facevid.pth'
    keypoint_detector, he_estimator, generator = set_face_vid(device, load_path, config=config)
    kp_detector_ddp = DDP(keypoint_detector, device_ids=[rank], find_unused_parameters=True)
    he_estimator_ddp = DDP(he_estimator, device_ids=[rank], find_unused_parameters=True)
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)

    # set fomm
    load_path_fomm = 'modules/FOMM/ckpt/FOMM.pth'
    config_path_fomm = 'config/vox-256_fomm.yaml'
    kp_detector_fomm, generator_fomm = set_fomm(device, load_path_fomm, config_path_fomm)
    kp_detector_fomm_ddp = DDP(kp_detector_fomm, device_ids=[rank], find_unused_parameters=True)
    generator_fomm_ddp = DDP(generator_fomm, device_ids=[rank], find_unused_parameters=True)

    # set deca
    deca = DECA()
    deca.eval()
    deca_ddp = DDP(deca, device_ids=[rank], find_unused_parameters=True)

    torch.cuda.empty_cache()

    dataset = FramesDataset(is_train=1, **config['dataset_params'])
    dataset = DatasetRepeater(dataset, 75)
    dataloader, CHANNELS = datasets.get_dataset_distributed_(
        dataset,
        world_size,
        rank,
        config['train_params']['batch_size']
    )

    history_best_psnr = 0
    history_best_ssim = 0

    for epoch in range(opt.n_epochs):

        test_noise = torch.randn(1, 3, 256, 256).to(device, non_blocking=True)

        noise_list = []
        for i in range(200):
            noise_list.append(torch.randn(1, 3, 256, 256).to(device, non_blocking=True))

        source_image = imageio.imread('sup-mat/source.png')
        driving_image = imageio.imread('sup-mat/driving.png')
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_image = resize(driving_image, (256, 256))[..., :3]
        s = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device, non_blocking=True)
        d = torch.tensor(driving_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device, non_blocking=True)

        source_image_r = imageio.imread('sup-mat/source_r.png')
        driving_image_r = imageio.imread('sup-mat/driving_r.png')
        source_image_r = resize(source_image_r, (256, 256))[..., :3]
        driving_image_r = resize(driving_image_r, (256, 256))[..., :3]
        s_r = torch.tensor(source_image_r[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device, non_blocking=True)
        d_r = torch.tensor(driving_image_r[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device, non_blocking=True)

        condition = get_trans_img(source_img=s, driving_img=d, keypoint_detector=kp_detector_ddp, he_estimator=he_estimator_ddp, generator=generator_ddp)
        condition_r = get_trans_img_fomm(source_img=s_r, driving_img=d_r, keypoint_detector=kp_detector_fomm_ddp, generator=generator_fomm_ddp)

        diff_condition = diffusion_condition(s=s, d=d, condition=condition, deca=deca_ddp, train=False)
        diff_condition_r = diffusion_condition(s=s_r, d=d_r, condition=condition_r, deca=deca_ddp, train=False)

        img = diffusion_ddp.module.refer(test_noise, condition=diff_condition, noise_list=noise_list)
        img_recon = diffusion_ddp.module.refer(test_noise, condition=diff_condition_r, noise_list=noise_list)

        img = torch.cat((img, condition['prediction'], s, d), dim=0)
        img_recon = torch.cat((img_recon, condition_r['prediction'], s_r, d_r), dim=0)

        if rank == 0:       
            save_image(img[:18], os.path.join(opt.output_dir, "%06d_reen.png" % epoch), nrow=5, normalize=False,
                    range=(-1, 1))
            save_image(img_recon[:18], os.path.join(opt.output_dir, "%06d_recon.png" % epoch), nrow=5, normalize=False,
                    range=(-1, 1))

        if rank == 0:
            log_file = open(os.path.join(opt.output_dir, 'log.txt'), 'a')
            l1 = torch.abs(img_recon[10:11, ...] - d_r).mean()
            diff_r = save_255_image(img_recon[10:11, ...], None)
            d_v = save_255_image(d_r, None)
            log_str = f'l1: {l1}; psnr: {peak_signal_noise_ratio(diff_r, d_v)}; ssim: {structural_similarity(diff_r, d_v, multichannel=True)}'
            if peak_signal_noise_ratio(diff_r, d_v) > history_best_psnr and structural_similarity(diff_r, d_v, multichannel=True) > history_best_ssim:
                history_best_psnr = peak_signal_noise_ratio(diff_r, d_v)
                history_best_ssim = structural_similarity(diff_r, d_v, multichannel=True)
                torch.save(diffusion_ddp.module.state_dict(), os.path.join(opt.output_dir, f'diffusion_generator{history_best_psnr}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(opt.output_dir, f'optimizer{history_best_psnr}.pth'))
            print(log_str, file=log_file)
            log_file.flush()
        
        for i, x in enumerate(dataloader):
            s = x['source'].to(device, non_blocking=True)
            d = x['driving'].to(device, non_blocking=True)

            x_in = {}
            x_in['real'] = d

            condition = get_trans_img_fomm(source_img=s, driving_img=d, keypoint_detector=kp_detector_fomm_ddp, generator=generator_fomm_ddp)
            x_in['condition'] = diffusion_condition(s=s, d=d, condition=condition, deca=deca_ddp)

            diffusion_ddp.train()

            for j in range(1):
                loss_dict = diffusion_ddp(x_in)
                loss = loss_dict['norm'] + loss_dict['color']
                if rank == 0 and j % 5 == 0 and i % 10 == 0:
                    for key in loss_dict.keys():
                        loss_ = loss_dict[key]
                        print(f'epoch {epoch} iter {len(dataloader)} / {i} repeat {j} warping {key} loss is {loss_.item()}')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if rank == 0:
            torch.save(diffusion_ddp.module.state_dict(), os.path.join(opt.output_dir, 'diffusion_generator.pth'))
            torch.save(optimizer.state_dict(), os.path.join(opt.output_dir, 'optimizer.pth'))
    
            log_file.close()
