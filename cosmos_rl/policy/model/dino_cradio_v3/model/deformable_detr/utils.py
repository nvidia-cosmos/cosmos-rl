import torch


def load_pretrained_weights(pretrained_path, parser=None):
    """To get over pytorch lightning module in the checkpoint state_dict.

    Args:
        pretrained_path (str): path to the pretrained model.
        parser (function): function to parse the state dict for a custom model.
    """
    temp = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "pytorch-lightning_version" not in temp and parser is not None:
        temp["state_dict"] = parser(temp)

    # for loading pretrained I3D weights released on
    # https://github.com/piergiaj/pytorch-i3d
    if "state_dict" not in temp:
        return temp

    state_dict = {}
    for key, value in list(temp["state_dict"].items()):
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        elif key.startswith("backbone."):
            # MMLab compatible weight loading
            new_key = key[9:]
            state_dict[new_key] = value
        elif key.startswith("model."):
            # MAE compatible weight loading
            new_key = key[len("model.") :]
            state_dict[new_key] = value
        elif key.startswith("ema_"):
            # Do not include ema params from MMLab
            continue
        else:
            state_dict[key] = value

    return state_dict


def _max_by_axis(the_list):
    """Get maximum image shape for padding."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def tensor_from_tensor_list(tensor_list, targets):
    """Convert list of tensors with different size to fixed resolution.

    The final size is determined by largest height and width.
    In theory, the batch could become [3, 1333, 1333] on dataset with different aspect ratio, e.g. COCO
    A fourth channel dimension is the mask region in which 0 represents the actual image and 1 means the padded region.
    This is to give size information to the transformer archicture. If transform-padding is applied,
    then only the pre-padded regions gets mask value of 1.

    Args:
        tensor_list (List[Tensor]): list of image tensors
        targets (List[dict]): list of labels that contain the size information

    Returns:
        tensors (torch.Tensor): list of image tensors in shape of (B, 4, H, W)
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        temp_tensors = torch.zeros((b, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        tensors = torch.concat((temp_tensors, mask), 1)
        for img, target, pad_img in zip(tensor_list, targets, tensors):
            # Get original image size before transform-padding
            # If no transform-padding has been applied,
            # then height == img.shape[1] and width == img.shape[2]
            actual_height, actual_width = target["size"]
            pad_img[: img.shape[0], :actual_height, :actual_width].copy_(
                img[:, :actual_height, :actual_width]
            )
            pad_img[c, :actual_height, :actual_width] = (
                0  # set zeros for mask in non-padded area
            )
    else:
        raise ValueError("Channel size other than 3 is not supported")
    return tensors


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
