"""
    Implementation of loss functions for the project
"""

import torch

class DiceLoss(torch.nn.Module):
    """
        Implements dice loss as described in 
            https://arxiv.org/abs/1807.10097
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        intersection = pred * target
        intersection = intersection.sum(-1).sum(-1)

        domain = pred ** 2 + target ** 2
        # Was running into nan issues, and added 1e-8 to prevent a division near
        # zero.  
        domain = domain.sum(-1).sum(-1) + 1e-8

        # Score for each class. Should work well for 
        dice_score = 2 * intersection / domain

        loss = 1 - dice_score.mean()
        return loss