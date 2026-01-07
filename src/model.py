import torch
import torch.nn as nn
from torchvision import models
from . import config

def build_model(num_classes=2, pretrained=True):
    """
    Builds a MobileNetV2 model for transfer learning.
    MobileNetV2 is efficient for CPU/Edge inference.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze feature extractor layers to speed up training
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier head
    # MobileNetV2 classifier is a Sequential block: (0): Dropout, (1): Linear
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model

if __name__ == "__main__":
    # Test model shape
    model = build_model()
    print("Model built successfully")
