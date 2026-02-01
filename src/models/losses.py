import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch.nn import BCEWithLogitsLoss

class FocalLossClassification(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Weighting factor for each class (similar to weight in CrossEntropy)
        gamma: Focusing parameter. Higher = more focus on hard samples.
        """
        super(FocalLossClassification, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate the probability of the correct class (pt)
        pt = torch.exp(-ce_loss)
        
        # Apply the focal weighting: (1-pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(mode='binary')

    def forward(self, pred, target):
        return 0.3 * self.bce(pred, target) + 0.7 * self.dice(pred, target)
    
class HybridFocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode='binary', from_logits=True)
    
    def focal_loss_with_logits(self, logits, targets, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.mean()

    def forward(self, pred, target):
        target = target.to(pred.dtype)

        dice_loss = self.dice(pred, target)
        focal_loss = self.focal_loss_with_logits(pred, target)
        
        return 0.7 * dice_loss + 0.3 * focal_loss
    
class HybridLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return (0.8 * self.dice(pred, target)) + (0.2 * self.bce(pred, target))