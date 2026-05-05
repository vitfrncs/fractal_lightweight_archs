"""
Salva logits e probabilidades de todas as amostras de validação e teste
em arquivos .npy, organizados por seed / backbone / dataset_type / split.

"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
from utils import DEVICE

def salvar_saidas(model, loader, seed,backbone: str, dataset_type, split = "val", 
                  base_dir = "../outputs/logits"):
    """
    Salva logits, probabilidades e rótulos.
    slipt: "val" ou "test"
        """
    model.eval()
    all_logits, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            # loader com 2 itens (img, label) ou 3 (orig, rec, label)
            if len(batch) == 3:
                imgs_orig, imgs_rec, labels = batch
                inputs = imgs_orig if dataset_type == "originais" else imgs_rec
            else:
                inputs, labels = batch
 
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs  = F.softmax(logits, dim=1)
 
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
 
    save_path = os.path.join(base_dir, str(seed), f"{backbone}_{dataset_type}_{split}")
    os.makedirs(save_path, exist_ok=True)
 
    np.save(os.path.join(save_path, "logits.npy"), np.concatenate(all_logits))
    np.save(os.path.join(save_path, "probs.npy"),  np.concatenate(all_probs))
    np.save(os.path.join(save_path, "labels.npy"), np.concatenate(all_labels))
 
    print(f"Saídas salvas em {save_path}")

def salvar_saidas_todos(seeds, results, val_loaders, test_loader,
                        base_dir="../outputs/logits"):
    """Salva logits/probs de validação e teste para todos os modelos.
    """
    config = [
        ("mobnet_orig",       "mobilenet",       "originais"),
        ("mobnet_recplot",    "mobilenet",       "F-RecPlot"),
        ("effnet_orig",       "efficientnet_b0", "originais"),
        ("effnet_recplot",    "efficientnet_b0", "F-RecPlot"),
        ("ghostnet_orig",     "ghostnet",        "originais"),
        ("ghostnet_recplot",  "ghostnet",        "F-RecPlot"),
    ]
 
    for seed in seeds:
        for chave, backbone, dataset_type in config:
            model = results[seed].get(chave)
            if model is None:
                print(f"Modelo {chave} seed {seed} não encontrado, pulando.")
                continue
 
            # validação
            if seed in val_loaders and (backbone, dataset_type) in val_loaders[seed]:
                salvar_saidas(
                    model, val_loaders[seed][(backbone, dataset_type)],
                    seed, backbone, dataset_type, split="val",
                    base_dir=base_dir,
                )
 
            # teste
            salvar_saidas(
                model, test_loader,
                seed, backbone, dataset_type, split="test",
                base_dir=base_dir,
            )
 



