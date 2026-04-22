import torch

def compute_iou(pred, target, num_classes=23):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    return torch.mean(torch.stack(ious))


def compute_dice(pred, target, num_classes=23):
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        dice = (2 * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-6)

        dices.append(dice)

    return torch.mean(torch.stack(dices))