import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedBrierFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, brier_weight=1.0, focal_weight=2.0, gamma=2.0):
        super(CombinedBrierFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.brier_weight = brier_weight
        self.focal_weight = focal_weight
    
    def brier_score_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
    
    def focal_loss(self, y_true, y_pred):
        # Focal loss for binary classification
        epsilon = 1e-9  # To avoid log(0)
        y_pred = y_pred.clamp(min=epsilon, max=1 - epsilon)  # Ensure numerical stability
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_loss = - alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)
        return torch.mean(focal_loss)
    
    def forward(self, y_pred, y_true):
        brier = self.brier_score_loss(y_true, y_pred)
        focal = self.focal_loss(y_true, y_pred)
        return self.brier_weight * brier + self.focal_weight * focal
