from dataset import EnsembleTestDataset
from metrics import metrics_to_csv
from save_outputs import salvar_saidas_todos
from gradcam import *
from efficiency import tabela_eficiencia
from utils import *
from dataset import *
from train import *
from model import *

import torch
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

N_GRADCAM = 30

DATASETS = [
    {
        "nome":         "displasia",
        "dataset_path": "../../../datasets/dataset_displasia/treino_e_validacao",
        "test_dir":     "../../../datasets/dataset_displasia/teste",
        "classes":      ["healthy", "severe"],
    },
#    {
#        "nome":         "pulmao",
#        "dataset_path": "../../../datasets/dataset_pulmao/treino_e_validacao",
#        "test_dir":     "../../../datasets/dataset_pulmao/teste",
#        "classes":      ["aca_md", "nor", "scc_md"],
#    },
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

    csv_path   = f"../outputs/{nome}/resultados_testes.csv"
    plots_dir  = f"../outputs/{nome}/plots"
    logits_dir = f"../outputs/{nome}/logits"
    gradcam_dir= f"../outputs/{nome}/gradcam"
    efic_path  = f"../outputs/{nome}/eficiencia.csv"

    test_dataset = EnsembleTestDataset(TEST_DIR, classes, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Total de pares para teste: {len(test_dataset)}")

    # === TREINAMENTO ===
    results = train_seeds(
        seeds        = SEEDS,
        dataset_path = dataset_path,
        classes      = classes,
        dataset_nome = nome,
        backbones    = ["ghostnet", "convnextv2"],
        skip_existing= True,
    )

    # === CARREGAR TODOS OS MODELOS ===
    for seed in SEEDS:
        # caminhos agora incluem nome do dataset
        base = f"../outputs/models/{nome}/{seed}"
        results[seed]["mobnet_orig"]      = carregar_modelo("mobilenet",       num_classes, f"{base}/mobilenet_originais.pth")
        results[seed]["mobnet_recplot"]   = carregar_modelo("mobilenet",       num_classes, f"{base}/mobilenet_F-RecPlot.pth")
        results[seed]["effnet_orig"]      = carregar_modelo("efficientnet_b0", num_classes, f"{base}/efficientnet_b0_originais.pth")
        results[seed]["effnet_recplot"]   = carregar_modelo("efficientnet_b0", num_classes, f"{base}/efficientnet_b0_F-RecPlot.pth")
        results[seed]["ghostnet_orig"]    = carregar_modelo("ghostnet",        num_classes, f"{base}/ghostnet_originais.pth")
        results[seed]["ghostnet_recplot"] = carregar_modelo("ghostnet",        num_classes, f"{base}/ghostnet_F-RecPlot.pth")
        results[seed]["convnext_orig"]    = carregar_modelo("convnextv2",      num_classes, f"{base}/convnextv2_originais.pth")
        results[seed]["convnext_recplot"] = carregar_modelo("convnextv2",      num_classes, f"{base}/convnextv2_F-RecPlot.pth")

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
        val_loaders = {},
        test_loader = test_loader,
        base_dir    = logits_dir,
    )

    # === GRAD-CAM ===
    seed_ref = SEEDS[0]
 
    for chave_orig, chave_rec, backbone in [
        ("ghostnet_orig", "ghostnet_recplot", "ghostnet"),
        ("convnext_orig", "convnext_recplot", "convnextv2"), # <-- Inclusão do pipeline Grad-CAM
    ]:
        model_orig = results[seed_ref].get(chave_orig)
        model_rec  = results[seed_ref].get(chave_rec)
 
        if model_orig is None:
            print(f"[SKIP Grad-CAM] {backbone}_orig não disponível")
            continue
 
        # Chamadas limpas sem parâmetro limitador (pega a pasta inteira de teste)
        gerar_gradcam_lote(
            model        = model_orig,
            backbone     = backbone,
            loader       = test_loader,
            class_names  = classes,
            split        = "test",
            base_dir     = gradcam_dir,
        )
 
        if model_rec is not None:
            gerar_gradcam_comparacao(
                model_orig   = model_orig,
                model_rec    = model_rec,
                backbone     = backbone,
                loader       = test_loader,
                class_names  = classes,
                split        = "test",
                base_dir     = gradcam_dir,
            )


    # === EFICIÊNCIA COMPUTACIONAL ===
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