"""
Geração de mapas Grad-CAM para visualização das regiões ativadas.

pip install grad-cam

"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

#gradcam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import DEVICE

#Camada alvo por backbone

def _get_target_layer(model, backbone: str):
    """Retorna a última camada convolucional para o Grad-CAM."""
    backbone = backbone.lower()
 
    if backbone == "mobilenet":
        return [model.features[-1][0]]          # Conv2d final do MobileNetV2
 
    elif backbone == "efficientnet_b0":
        return [model.features[-1][0]]          # Conv2d final do EfficientNet-B0
 
    elif backbone == "ghostnet":
        # No GhostNet do timm a última conv fica em conv_head
        return [model.conv_head]
 
    else:
        raise ValueError(
            f"Backbone '{backbone}' não mapeado. "
            "Adicione a camada alvo manualmente em _get_target_layer()."
        )


# Normalização inversa para visualização 
 
_inv_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229,       1/0.224,       1/0.225],
    )
])
 
def _tensor_to_rgb(tensor):
    """Converte tensor normalizado em array RGB float32 em [0, 1]."""
    img = _inv_transform(tensor.cpu()).permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1).astype(np.float32)

# Gradcam para uma imagem

def gerar_gradcam(model, backbone: str, img_tensor: torch.Tensor,
    label: int, save_path: str,):
    """Gera e salva o mapa Grad-CAM de uma única imagem."""
    target_layers = _get_target_layer(model, backbone)
    cam = GradCAM(model=model, target_layers=target_layers)
 
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    targets      = [ClassifierOutputTarget(label)]
 
    mask          = cam(input_tensor=input_tensor, targets=targets)
    rgb_image     = _tensor_to_rgb(img_tensor)
    visualization = show_cam_on_image(rgb_image, mask[0], use_rgb=True)
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(visualization).save(save_path)


# Gradcam para um dataloarder

def gerar_gradcam_lote(
    model,
    backbone: str,
    loader,
    class_names: list,
    split: str = "test",
    n_por_classe: int = 5,
    base_dir: str = "../outputs/gradcam",
):
    """Gera mapas Grad-CAM para as primeiras N imagens de cada classe.    base_dir:     Diretório raiz de saída.
    """
    contagem = {i: 0 for i in range(len(class_names))}
    total_alvo = n_por_classe * len(class_names)
    gerados = 0
 
    model.eval()
 
    for batch in loader:
        if gerados >= total_alvo:
            break
 
        # loader com 2 ou 3 itens
        if len(batch) == 3:
            imgs_orig, _, labels = batch
            imgs = imgs_orig
        else:
            imgs, labels = batch
 
        for i in range(imgs.size(0)):
            lbl = int(labels[i].item())
            if contagem[lbl] >= n_por_classe:
                continue
 
            idx   = contagem[lbl]
            fname = f"{class_names[lbl]}_{idx:03d}.png"
            path  = os.path.join(base_dir, backbone, split,
                                 class_names[lbl], fname)
 
            gerar_gradcam(
                model=model,
                backbone=backbone,
                img_tensor=imgs[i],
                label=lbl,
                save_path=path,
            )
 
            contagem[lbl] += 1
            gerados        += 1
 
            if gerados >= total_alvo:
                break
 
    print(f"Grad-CAM gerado: {gerados} imagens em {base_dir}/{backbone}/{split}/")
