"""
Loss functions for U-Net training.

Provides Dice loss, combined BCE + Dice loss, and a Dice coefficient
metric for validation monitoring.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross-Entropy and Dice loss.

    Balances pixel-wise classification accuracy (BCE) with region-level
    overlap (Dice), which helps manage class imbalance at cell boundaries.
    """

    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_dice_coefficient(logits, targets, threshold=0.5, smooth=1e-5):
    """
    Calculate the Dice coefficient for validation monitoring.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output (before sigmoid).
    targets : torch.Tensor
        Ground truth binary masks.
    threshold : float
        Binarisation threshold applied after sigmoid.
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()
