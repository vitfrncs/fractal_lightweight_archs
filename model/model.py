"""
Transformei em uma factory pra facilitar a adição de outros modelos.

Para adicionar outra arquitetura:
 -Criar arquivo em architectures com criar_nomeDaArc
 -Exportar no init de architectures
 -Mudar o registro desse arquivo aqui
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from utils import DEVICE
from architectures import (
    criar_mobilenet,
    criar_efficientnet_b0,
    criar_ghostnet,
)

# Registro de backbones ──────────────────────────────────────────
REGISTRO: dict = {
    "mobilenet":       criar_mobilenet,
    "efficientnet_b0": criar_efficientnet_b0,
    "ghostnet":        criar_ghostnet,
}

def criar_modelo(backbone, num_classes, pretrained = True):
    """Instancia o modelo pelo nome do backbone. 
    Retorna o modelo. 
    """
    key = backbone.lower()
    if key not in REGISTRO:
        opcoes = ", ".join(REGISTRO.keys())
        raise ValueError(
            f"Backbone '{backbone}' não encontrado. Opções: {opcoes}"
        )
    return REGISTRO[key](num_classes=num_classes, pretrained=pretrained)



def carregar_modelo(backbone, num_classes, path_weights):
    """Reconstroi modelo e carrega pesos salvos."""
    print(f"Carregando {backbone} de {path_weights}...")
    model = criar_modelo(backbone=backbone, num_classes=num_classes, pretrained=False)
 
    # Carrega os pesos treinados
    try:
        state = torch.load(path_weights, map_location=DEVICE)
        model.load_state_dict(state)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{path_weights}' não encontrado. Treine o modelo antes.")
        return None

    model.to(DEVICE)
    model.eval()
    return model
