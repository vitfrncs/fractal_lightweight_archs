import torch.nn as nn
import torchvision.models as models
 
 
def criar_mobilenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 com cabeça classificadora substituída."""
    model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
