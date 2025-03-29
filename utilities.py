from PIL import Image
import numpy as np
import scipy.ndimage
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

def kl_divergence_loss(y_pred, y_true):
    y_pred = F.log_softmax(y_pred, dim=1)
    y_true = F.softmax(y_true, dim=1)
    return F.kl_div(y_pred, y_true, reduction='batchmean')


def correlation_coefficient_loss(y_pred, y_true):
    y_pred_mean = torch.mean(y_pred, dim=(1, 2, 3), keepdim=True)
    y_true_mean = torch.mean(y_true, dim=(1, 2, 3), keepdim=True)
    y_pred_centered = y_pred - y_pred_mean
    y_true_centered = y_true - y_true_mean
    correlation = torch.sum(y_pred_centered * y_true_centered, dim=(1, 2, 3))
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=(1, 2, 3)) + 1e-6)
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=(1, 2, 3)) + 1e-6)
    return -correlation / (std_pred * std_true + 1e-6)


class CombinedLoss(nn.Module):
    def __init__(self, alpha1=0.1, alpha2=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true):
        l_kl = kl_divergence_loss(y_pred, y_true)
        l_cc = correlation_coefficient_loss(y_pred, y_true)
        loss = self.alpha1 * l_kl + l_cc
        return loss.mean()
