"""Mede a eficiência de cada modelo.

pip install thop

"""

import time
import torch
import pandas as pd
from thop import profile as thop_profile

from utils import DEVICE


def medir_eficiencia(
    model,
    backbone: str,
    input_size: tuple = (1, 3, 224, 224),
    n_repeticoes: int = 50,
) -> dict:
    """Mede parâmetros, FLOPs e latência média de inferência."""
    model.eval()
    dummy = torch.randn(*input_size).to(DEVICE)
 
    # parâmetros treináveis
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
 
    # FLOPs
    gflops = None
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    gflops  = round(macs * 2 / 1e9, 4)   # MACs × 2 ≈ FLOPs

    # latência de inferência 
    # aquecimento :P
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
 
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
 
    tempos = []
    with torch.no_grad():
        for _ in range(n_repeticoes):
            t0 = time.perf_counter()
            model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            tempos.append((time.perf_counter() - t0) * 1000)   # ms
 
    latencia_media = round(sum(tempos) / len(tempos) / input_size[0], 4)
 
    return {
        "backbone":             backbone,
        "params_M":             round(params / 1e6, 3),
        "GFLOPs":               gflops,
        "latencia_ms_por_img":  latencia_media,
    }


def tabela_eficiencia(modelos,csv_path= "../outputs/eficiencia.csv"):
    """Gera e salva tabela de eficiência para vários modelos."""
    rows = []
    for nome, model in modelos.items():
        if model is None:
            print(f"Modelo '{nome}' é None, pulando.")
            continue
        print(f"Medindo eficiência de {nome}...")
        rows.append(medir_eficiencia(model, backbone=nome))
 
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nTabela de eficiência salva em {csv_path}")
    print(df.to_string(index=False))
    return df
