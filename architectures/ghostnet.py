import torch.nn as nn
import timm # usando esse modelo, podemos usar pretreino

def criar_ghostnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """GhostNet-1.0x via timm, com cabeça classificadora substituída.
    """
    model = timm.create_model("ghostnet_100", pretrained=pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
 
