import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import ConvNextForImageClassification, BeitForImageClassification, ViTForImageClassification,AutoModelForImageClassification

from timm import create_model

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
class Finetuned_ConvNext(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = ConvNextForImageClassification.from_pretrained("ahsanjavid/convnext-tiny-finetuned-cifar10")
        self.backbone = model
        
    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(x)
        return x

class Finetuned_VIT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
        self.backbone = model
        
    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(x)
        return x


class ConvNextv2_large(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = ConvNextForImageClassification.from_pretrained("timm/convnextv2_large.fcmae_ft_in22k_in1k_384")
        self.backbone = model
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(self.classifier(x))
        return x
    
        
class Beit_base(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        self.backbone = model
        self.classifier = nn.Linear(21841, num_classes)
        
    def forward(self, x):
        x = self.backbone(x).logits
        x = F.sigmoid(self.classifier(x))
        return x

# size 256
class Maxvit_base(nn.Module):
    def __init__(self):
        super().__init__()
        model = create_model("maxvit_rmlp_tiny_rw_256", pretrained=True, num_classes=10)
        self.backbone = model
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.sigmoid(x)
        return x



