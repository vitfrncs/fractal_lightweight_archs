from dataset import EnsembleTestDataset
from metrics import metrics_to_csv
from save_outputs import salvar_saidas_todos
from gradcam import gerar_gradcam_lote
from efficiency import tabela_eficiencia
from utils import *
from dataset import *
from train import *
from model import *

import torch
from torch.utils.data import DataLoader


import sys, os
sys.path.insert(0, os.path.dirname(__file__))  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # architectures/

# Configuração dos datasets:

DATASETS = [
    {
        "nome":         "displasia",
        "dataset_path": "../datasets/dataset_displasia/treino_e_validacao",
        "test_dir":     "../datasets/dataset_displasia/teste",
        "classes":      ["healthy", "severe"],
    },
    {
        "nome":         "pulmao",
        "dataset_path": "../datasets/dataset_pulmao/treino_e_validacao",
        "test_dir":     "../datasets/dataset_pulmao/teste",
        "classes":      ["aca_md", "nor", "scc_md"],
    },
]



for ds in DATASETS:
    nome         = ds["nome"]
    dataset_path = ds["dataset_path"]
    TEST_DIR     = ds["test_dir"]
    classes      = ds["classes"]
    num_classes  = len(classes)
 
    print(f"\n{'='*60}")
    print(f"  DATASET: {nome.upper()}  |  classes: {classes}")
    print(f"{'='*60}")
 
    # caminhos de saída específicos por dataset
    csv_path  = f"../outputs/{nome}/resultados_testes.csv"
    plots_dir = f"../outputs/{nome}/plots"
    logits_dir= f"../outputs/{nome}/logits"
    gradcam_dir=f"../outputs/{nome}/gradcam"
    efic_path = f"../outputs/{nome}/eficiencia.csv"
 
    # Test loader
    test_dataset = EnsembleTestDataset(TEST_DIR, classes, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Total de pares para teste: {len(test_dataset)}")
 
    # === TREINAMENTO ===
    results = train_seeds(
        seeds        = SEEDS,
        dataset_path = dataset_path,
        classes      = classes,
        backbones    = ["ghostnet"],
        skip_existing= True,
    )
 
    # ====nCARREGAR TODOS OS MODELOS ====
    for seed in SEEDS:
        results[seed]["mobnet_orig"]      = carregar_modelo("mobilenet",       num_classes, f"../outputs/models/{seed}/mobilenet_originais.pth")
        results[seed]["mobnet_recplot"]   = carregar_modelo("mobilenet",       num_classes, f"../outputs/models/{seed}/mobilenet_F-RecPlot.pth")
        results[seed]["effnet_orig"]      = carregar_modelo("efficientnet_b0", num_classes, f"../outputs/models/{seed}/efficientnet_b0_originais.pth")
        results[seed]["effnet_recplot"]   = carregar_modelo("efficientnet_b0", num_classes, f"../outputs/models/{seed}/efficientnet_b0_F-RecPlot.pth")
        results[seed]["ghostnet_orig"]    = carregar_modelo("ghostnet",        num_classes, f"../outputs/models/{seed}/ghostnet_originais.pth")
        results[seed]["ghostnet_recplot"] = carregar_modelo("ghostnet",        num_classes, f"../outputs/models/{seed}/ghostnet_F-RecPlot.pth")
 
    # === MÉTRICAS DE TESTE + MATRIZES DE CONFUSÃO ===
    metrics_to_csv(
        seeds       = SEEDS,
        results     = results,
        test_loader = test_loader,
        class_names = classes,
        csv_path    = csv_path,
        plots_dir   = plots_dir,
    )
 
    # === SALVAR LOGITS E PROBABILIDADES ===
    salvar_saidas_todos(
        seeds       = SEEDS,
        results     = results,
        val_loaders = {},       # passe val_loaders preenchido se quiser salvar validação
        test_loader = test_loader,
        base_dir    = logits_dir,
    )
 
    # === GRAD-CAM ===
    seed_ref = SEEDS[0]
    for chave, backbone in [
        # ("mobnet_orig",   "mobilenet"),        # descomentar quando disponível
        # ("effnet_orig",   "efficientnet_b0"),  # descomentar quando disponível
        ("ghostnet_orig", "ghostnet"),
    ]:
        model = results[seed_ref].get(chave)
        if model is None:
            print(f"[SKIP Grad-CAM] {backbone} não disponível para {nome}")
            continue
        gerar_gradcam_lote(
            model        = model,
            backbone     = backbone,
            loader       = test_loader,
            class_names  = classes,
            split        = "test",
            n_por_classe = 5,
            base_dir     = gradcam_dir,
        )
 
    # === EFICIÊNCIA COMPUTACIONAL ====
    modelos_efic = {
        nome_backbone: results[seed_ref][chave]
        for nome_backbone, chave in [
            ("mobilenet",       "mobnet_orig"),
            ("efficientnet_b0", "effnet_orig"),
            ("ghostnet",        "ghostnet_orig"),
        ]
        if results[seed_ref].get(chave) is not None
    }
 
    if modelos_efic:
        tabela_eficiencia(modelos=modelos_efic, csv_path=efic_path)
    else:
        print(f"[SKIP Eficiência] Nenhum modelo disponível para {nome}")
