from turtle import forward
import torch
import torch.nn as nn
from .util import Block, Swish
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlockCondition(nn.Module):
    def __init__(self, dim, pre_dim,  dim_out, dropout=0, norm_groups=16):
        super().__init__()
        # self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block1 = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            pre_dim, dim_out, 1) if pre_dim != dim_out else nn.Identity()
        
    def forward(self, pre_image, up_features):
        _, _, h, w = pre_image.size()
        _, _, h_old, w_old = up_features.size()
        if h_old != h or w_old != w:
            up_features = F.interpolate(up_features, size=(h, w), mode='bilinear')
        out = self.block2(self.block1(up_features))
        return out + 0.4 * self.res_conv(pre_image)


class MLPCondition(nn.Module):
    def __init__(self, time_dim, condition_dim, inner_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_dim, inner_dim)
        )
        self.condition = nn.Sequential(
            Swish(),
            nn.Linear(condition_dim+inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.Sigmoid()
        )
        self.inner_dim = inner_dim
    
    def forward(self, t, attribute):
        """
        attribute is the distance of exp, pose between d and s
        """
        t = self.time_mlp(t)
        attri = torch.cat((attribute, t), dim=1)
        return self.condition(attri) + 0.4 * F.adaptive_max_pool1d(attribute.unsqueeze(1), self.inner_dim).squeeze(1)


class condition_net(nn.Module):
    def __init__(self, time_dim, condition_dim, in_channel=3, inner_channel=32, out_channel=3):
        super().__init__()
        self.pre_conv_color = nn.Sequential(
                              nn.Conv2d(in_channel, inner_channel // 2, 7, padding=3),
                              nn.ReLU(),
                              ResnetBlock(inner_channel // 2, inner_channel // 2, norm_groups=16),
                              nn.ReLU(),
                              ResnetBlock(inner_channel // 2, inner_channel // 2, norm_groups=16),
                              Swish())
        self.deal_cond = ResnetBlockCondition(dim=in_channel, pre_dim=inner_channel // 2, 
                                              dim_out=inner_channel // 2)
        self.motion_cond = MLPCondition(time_dim=time_dim, condition_dim=condition_dim, inner_dim=inner_channel // 2)
        self.inner_channel = inner_channel
        self.resnet = nn.Sequential(
                            ResnetBlock(inner_channel, inner_channel),
                            nn.ReLU(),
                            ResnetBlock(inner_channel, inner_channel),
                            nn.ReLU(),
                            ResnetBlock(inner_channel, inner_channel),
                            nn.ReLU(),
                            nn.Conv2d(inner_channel, out_channel, 7, padding=3),
                            nn.Sigmoid())

    def motion_weight(self, w, level, K):
        wi = (K - level) / K * torch.exp(w - 0.3) + level / K * torch.exp(- w + 0.3)
        return wi

    def forward(self, warp_source, color_cond, attribute, t):
        """
        warp_source include multi resolution of generated_images
        color_cond include down sample driving_images and generated_images
        """
        generated_img = color_cond['generated']
        driving_img = color_cond['driving']
        generated_img = self.pre_conv_color(generated_img)
        driving_img = self.pre_conv_color(driving_img)
        feature_loss = torch.abs(generated_img - driving_img).mean()
        cond_list = []
        motion_w = self.motion_cond(t=t, attribute=attribute)
        for i in range(0, len(warp_source)):
            motion_wi = self.motion_weight(motion_w, i, len(warp_source) - 1)
            cond = self.deal_cond(pre_image=driving_img, up_features=warp_source[i]) * motion_wi[:, :, None, None]
            cond_list.append(cond)
        cond = cond_list[0] + cond_list[1] + cond_list[2]
        out = torch.cat((cond, generated_img + 0.2 * driving_img), dim=1)
        return self.resnet(out), feature_loss
