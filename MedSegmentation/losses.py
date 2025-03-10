from __future__ import print_function, division

import torch


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def jaccard_loss(prediction, target):
    """Calculating the jaccard index
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = abs((i_flat * t_flat)).sum()

    return 1 - ((intersection + smooth) / (i_flat.sum() + t_flat.sum() - intersection + smooth))


def calc_loss(prediction, target, bce_weight=0.1, dice_weight=0.45, jaccard_weight=0.45):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = torch.nn.BCELoss()
    bce = bce(prediction, target)  # F.binary_cross_entropy_with_logits(prediction, target)
    # prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    jac = jaccard_loss(prediction, target)

    loss = bce * bce_weight + dice * dice_weight + jac * jaccard_weight

    return loss
