import torch
import torch.nn as nn
import torchvision.models as models

def criar_convnextv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instancia o modelo ConvNeXtV2 Tiny e ajusta a camada de classificação."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    
    # substitui head para o número de classes que temos. 
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model