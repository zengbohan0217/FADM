import torch


def get_dataset_distributed_(_dataset, world_size, rank, batch_size, **kwargs):

    sampler = torch.utils.data.distributed.DistributedSampler(
        _dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        _dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=16,
        persistent_workers=True,
    )

    return dataloader, 3
