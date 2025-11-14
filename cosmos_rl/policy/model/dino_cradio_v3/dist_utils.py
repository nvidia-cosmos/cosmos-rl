import torch.distributed as dist


def is_dist_avail_and_initialized():
    """Check if DDP is initialized."""
    is_dist = True
    if not dist.is_available():
        is_dist = False
    else:
        is_dist = dist.is_initialized() or False
    return is_dist


def get_global_rank():
    """Get global rank."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get world size."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
