import torch
import torch.nn as nn

class SmoothBCELoss(nn.Module):
    def __init__(self, epsilon=0.05):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.epsilon = epsilon
    
    def forward(self, targets, labels):
        smooth_labels = labels.detach().clone()
        for i, x in enumerate(smooth_labels):
            smooth_labels[i] = torch.where(x==1.0, 1.0-self.epsilon, self.epsilon)
        smoothed_loss = self.criterion(targets, smooth_labels)
        return smoothed_loss
    
    
    