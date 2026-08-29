import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.lossall import DifferentiableBoundaryLoss

class FocalBCELoss(nn.Module):
    def __init__(self, focal_gamma=2.0):
        super().__init__()
        self.focal_gamma = focal_gamma

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: [B, C, H, W] logits
            target: [B, C, H, W] target probabilities/one-hot
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        focal_loss = ((1 - pt) ** self.focal_gamma) * bce_loss
        return focal_loss.mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs, target):
        """
        Args:
            pred_probs: [B, C, H, W] probabilities
            target: [B, C, H, W] target probabilities/one-hot
        """
        intersection = (pred_probs * target).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice_scores).mean()

class CalculusLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_bce = FocalBCELoss(focal_gamma=2.0)
        self.soft_dice = SoftDiceLoss(smooth=1.0)
        self.boundary_loss = DifferentiableBoundaryLoss()
        
        # Loss weights
        self.bce_coarse = 1.5
        self.dice_coarse = 1.0
        self.conf_weight = 0.5
        self.bce_refine = 1.0
        self.dice_refine = 1.0
        self.boundary_refine = 0.3

    def forward(self, sam_masks, refined_masks, confidence, gt_masks):
        """
        Args:
            sam_masks: [B, 2, H, W] (coarse SAM logits)
            refined_masks: [B, 2, H, W] (refined logits)
            confidence: [B, num_queries] (confidence scores)
            gt_masks: [B, 2, H, W] (one-hot: channel 0=bg, channel 1=calculus)
            
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual loss components
        """
        B, num_queries = confidence.shape
        
        # Determine confidence target (1.0 if any calculus pixel exists, 0.0 otherwise)
        has_calculus = (gt_masks[:, 1].sum(dim=(1, 2)) > 0).float() # [B]
        conf_target = has_calculus.unsqueeze(1).expand(-1, num_queries) # [B, num_queries]
        
        # --- Stage 1: Coarse SAM masks ---
        sam_probs = torch.sigmoid(sam_masks)
        loss_bce_coarse = self.focal_bce(sam_masks, gt_masks)
        loss_dice_coarse = self.soft_dice(sam_probs, gt_masks)
        loss_conf = F.binary_cross_entropy_with_logits(confidence, conf_target, reduction='mean')
        
        # --- Stage 2: Refined masks ---
        refined_probs = torch.sigmoid(refined_masks)
        loss_bce_refine = self.focal_bce(refined_masks, gt_masks)
        loss_dice_refine = self.soft_dice(refined_probs, gt_masks)
        loss_boundary = self.boundary_loss(refined_probs, gt_masks)
        
        # --- Total Loss ---
        total_loss = (
            self.bce_coarse * loss_bce_coarse +
            self.dice_coarse * loss_dice_coarse +
            self.conf_weight * loss_conf +
            self.bce_refine * loss_bce_refine +
            self.dice_refine * loss_dice_refine +
            self.boundary_refine * loss_boundary
        )
        
        loss_dict = {
            'loss_bce_coarse': loss_bce_coarse.item(),
            'loss_dice_coarse': loss_dice_coarse.item(),
            'loss_conf': loss_conf.item(),
            'loss_bce_refine': loss_bce_refine.item(),
            'loss_dice_refine': loss_dice_refine.item(),
            'loss_boundary': loss_boundary.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
