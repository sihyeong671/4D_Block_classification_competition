import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import ConvNextForImageClassification

class ConvNext_xlarge(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-xlarge-384-22k-1k")
        self.backbone = model
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(self.classifier(x))
        return x

