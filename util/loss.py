import torch
import torch.nn as nn
import numpy as np


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
    

def sigmoid(x):
  return (1/(1+np.exp(-x)))


class CDB_loss(nn.Module):
  
    def __init__(self, class_difficulty, tau='dynamic', reduction='mean'):
        
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.reduction = reduction
        self.loss = nn.BCELoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda()
        

    def forward(self, input, target):

        return self.loss(input, target)
    
    