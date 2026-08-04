"""
Geração de mapas Grad-CAM para visualização das regiões ativadas.

pip install grad-cam
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import DEVICE

import torch.nn as nn

def get_target_layers(model):
    """Seleciona automaticamente uma camada adequada com base na arquitetura.
    """
    
    # percorre o grafo dos modelos para achar a última camada convolucional
    last_conv_name = None
    last_conv_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
            last_conv_module = module

    if last_conv_module is not None:
        return [last_conv_module]

    # Fallback para arquiteturas baseadas em transformers 
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return [model.blocks[-1]]

    # Fallback para modelos com 'features' como container genérico
    if hasattr(model, "features") and len(list(model.features.children())) > 0:
        return [model.features[-1]]

    # 4.Se nenhum dos ifs funcionar, retorna o último submódulo direto do modelo.
    children = list(model.children())
    if children:
        return [children[-1]]

    return [model]



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



#  Wrapper — instancia GradCAM uma vez por modelo


class GradCAMWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.to(DEVICE)

        self.target_layers = get_target_layers(model)
        print(f"[GradCAM] Camada selecionada: {self.target_layers[0]}")

        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def get_cam_overlay(self, img_tensor: torch.Tensor, label: int) -> np.ndarray:
        """Retorna o overlay Grad-CAM como array uint8 (H, W, 3)."""
        input_tensor  = img_tensor.unsqueeze(0).to(DEVICE)
        targets       = [ClassifierOutputTarget(label)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        rgb_image     = _tensor_to_rgb(img_tensor)
        overlay       = show_cam_on_image(rgb_image, grayscale_cam[0], use_rgb=True)
        return overlay  # uint8 (H, W, 3)

    def generate(self, img_tensor: torch.Tensor, label: int, save_path: str):
        """Gera e salva o mapa Grad-CAM de uma única imagem."""
        overlay = self.get_cam_overlay(img_tensor, label)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(overlay).save(save_path)



# Comparação lado a lado: original vs recplot


def _add_label(img_array: np.ndarray, text: str, font_size: int = 14) -> Image.Image:
    """Adiciona legenda abaixo da imagem."""
    img = Image.fromarray(img_array)
    W, H = img.size
    label_h = font_size + 8
    canvas = Image.new("RGB", (W, H + label_h), (30, 30, 30))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, H + 2), text, fill=(255, 255, 255), font=font)
    return canvas


def _make_comparison(
    orig_tensor, rec_tensor,
    wrapper_orig, wrapper_rec,
    label_orig: int,
    label_rec: int,
) -> Image.Image:
    """
    Monta uma imagem comparativa com 4 painéis:
        Original | Grad-CAM Original | F-RecPlot | Grad-CAM F-RecPlot
    """
    orig_rgb = (_tensor_to_rgb(orig_tensor) * 255).astype(np.uint8)
    rec_rgb  = (_tensor_to_rgb(rec_tensor)  * 255).astype(np.uint8)

    cam_orig = wrapper_orig.get_cam_overlay(orig_tensor, label_orig)
    cam_rec  = wrapper_rec.get_cam_overlay(rec_tensor,  label_rec)

    panels = [
        _add_label(orig_rgb,  "Original"),
        _add_label(cam_orig,  "GradCAM Original"),
        _add_label(rec_rgb,   "F-RecPlot"),
        _add_label(cam_rec,   "GradCAM F-RecPlot"),
    ]

    total_w = sum(p.width for p in panels)
    max_h   = max(p.height for p in panels)
    strip   = Image.new("RGB", (total_w, max_h), (30, 30, 30))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width

    return strip



# Geração em lote — individual


def gerar_gradcam_lote(
    model,
    backbone: str,
    loader,
    class_names: list,
    split: str = "test",
    base_dir: str = "../outputs/gradcam",
):
    """Gera mapas Grad-CAM individuais para ABSOLUTAMENTE TODAS as imagens do loader."""
    wrapper  = GradCAMWrapper(model)
    contagem = {i: 0 for i in range(len(class_names))}
    gerados  = 0
    acertos  = 0
    erros    = 0

    for batch in loader:
        if len(batch) == 3:
            imgs, _, labels = batch
        else:
            imgs, labels = batch

        imgs_dev = imgs.to(DEVICE)
        with torch.no_grad():
            outputs = wrapper.model(imgs_dev)
            preds   = outputs.argmax(dim=1)

        for i in range(imgs.size(0)):
            lbl_real = int(labels[i].item())
            lbl_pred = int(preds[i].item())
            correto  = lbl_real == lbl_pred
            subdir   = "correct" if correto else "incorrect"

            idx = contagem[lbl_real]

            fname_pred = f"{class_names[lbl_real]}_{idx:03d}_pred-{class_names[lbl_pred]}.png"
            path_pred  = os.path.join(
                base_dir, backbone, split, subdir, class_names[lbl_real], fname_pred
            )
            wrapper.generate(img_tensor=imgs[i], label=lbl_pred, save_path=path_pred)
            gerados += 1

            if not correto:
                fname_true = f"{class_names[lbl_real]}_{idx:03d}_true-{class_names[lbl_real]}.png"
                path_true  = os.path.join(
                    base_dir, backbone, split, subdir, class_names[lbl_real], fname_true
                )
                wrapper.generate(img_tensor=imgs[i], label=lbl_real, save_path=path_true)
                gerados += 1
                erros += 1
            else:
                acertos += 1

            contagem[lbl_real] += 1

    print(
        f"Grad-CAM individual completo: {gerados} imagens salvas em "
        f"{base_dir}/{backbone}/{split}/ ({acertos} acertos, {erros} erros)"
    )



# Geração em lote — comparação original vs recplot


def gerar_gradcam_comparacao(
    model_orig,
    model_rec,
    backbone: str,
    loader,
    class_names: list,
    split: str = "test",
    base_dir: str = "../outputs/gradcam",
):
    """Gera imagens comparativas lado a lado para ABSOLUTAMENTE TODAS as imagens do loader."""
    wrapper_orig = GradCAMWrapper(model_orig)
    wrapper_rec  = GradCAMWrapper(model_rec)

    contagem = {i: 0 for i in range(len(class_names))}
    gerados  = 0

    for batch in loader:
        if len(batch) != 3:
            raise ValueError("Requer EnsembleTestDataset (batch com 3 itens: orig, rec, label).")

        imgs_orig, imgs_rec, labels = batch
        imgs_orig_dev = imgs_orig.to(DEVICE)
        imgs_rec_dev  = imgs_rec.to(DEVICE)

        with torch.no_grad():
            preds_orig = wrapper_orig.model(imgs_orig_dev).argmax(dim=1)
            preds_rec  = wrapper_rec.model(imgs_rec_dev).argmax(dim=1)

        for i in range(imgs_orig.size(0)):
            lbl_real      = int(labels[i].item())
            lbl_pred_orig = int(preds_orig[i].item())
            lbl_pred_rec  = int(preds_rec[i].item())

            idx = contagem[lbl_real]
            fname = (
                f"{class_names[lbl_real]}_{idx:03d}"
                f"_predOrig-{class_names[lbl_pred_orig]}"
                f"_predRec-{class_names[lbl_pred_rec]}.png"
            )
            path = os.path.join(base_dir, f"{backbone}_comparacao", split, class_names[lbl_real], fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            comparison = _make_comparison(
                orig_tensor  = imgs_orig[i],
                rec_tensor   = imgs_rec[i],
                wrapper_orig = wrapper_orig,
                wrapper_rec  = wrapper_rec,
                label_orig   = lbl_pred_orig,
                label_rec    = lbl_pred_rec,
            )
            comparison.save(path)

            contagem[lbl_real] += 1
            gerados += 1

    out_dir = os.path.join(base_dir, f"{backbone}_comparacao", split)
    print(f"Grad-CAM comparação completa: {gerados} imagens salvas em {out_dir}/")
