"""
Utilities for bounding box manipulation and GIoU.
"""

import torch

from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """Convert cxcywh format to xyxy."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert xyxy format to cxcywh."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """Calculate box IoU.

        Args:
        boxes1 (torch.Tensor): boxes 1.
        boxes2 (torch.Tensor): boxes 2.

    Returns:
        iou (torch.Tensor): IoU values.
        union (torch.Tensor): Union values.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/.

    The boxes should be in [x0, y0, x1, y1] format.

    Args:
        boxes1 (torch.Tensor): boxes 1.
        boxes2 (torch.Tensor): boxes 2.

    Returns:
        A [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2).

    Raises:
        Degenerate boxes gives inf / nan results, so do an early check.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
