"""Para exportar cada backbone de forma isolada, sem dependência 
do resto do projeto."""

from .mobilenet import criar_mobilenet
from .efficientnet import criar_efficientnet_b0
from .ghostnet import criar_ghostnet
from .convnextv2 import criar_convnextv2
 
__all__ = [
    "criar_mobilenet",
    "criar_efficientnet_b0",
    "criar_ghostnet",
    "criar_convnextv2",
]
