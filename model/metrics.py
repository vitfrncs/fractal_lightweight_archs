import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from utils import DEVICE


def mostrar_metricas(nome_cenario, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    specificities = []

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)

    specificity = np.mean(specificities)

    return {
        "acc": acc,
        "f1_macro": f1,
        "recall_macro": recall,
        "specificity_macro": specificity
    }

def salvar_matriz_confusao(y_true, y_pred, class_names, save_path: str):
    """Salva a matriz de confusão em png."""
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(os.path.basename(save_path).replace(".png", ""))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matriz de confusão salva em {save_path}")


def metrics_to_csv(seeds, results, test_loader, class_names,
                   csv_path="outputs/resultados_testes.csv", plots_dir="outputs/plots"):
    """Avalia redes individuais e ensembles de forma otimizada no conjunto de teste."""
    all_results = []

    # 1. Definição clara e única de cada modelo e seu tipo de entrada
    model_meta = {
        "MobileNet_Original":  {"key": "mobnet_orig",      "input": "orig"},
        "MobileNet_RecPlot":   {"key": "mobnet_recplot",   "input": "rec"},
        "EffNet_Original":     {"key": "effnet_orig",      "input": "orig"},
        "EffNet_RecPlot":      {"key": "effnet_recplot",   "input": "rec"},
        "GhostNet_Original":   {"key": "ghostnet_orig",    "input": "orig"},
        "GhostNet_RecPlot":    {"key": "ghostnet_recplot", "input": "rec"},
        "ConvNeXtV2_Original": {"key": "convnext_orig",    "input": "orig"},
        "ConvNeXtV2_RecPlot":  {"key": "convnext_recplot", "input": "rec"},
    }

    # 2. Definição dos Ensembles Reais (Combinações sem repetição)
    ensembles_config = [
        ("MobileNet_Original",  "MobileNet_RecPlot"),
        ("MobileNet_Original",  "EffNet_Original"),
        ("MobileNet_Original",  "EffNet_RecPlot"),
        ("MobileNet_Original",  "GhostNet_Original"),
        ("MobileNet_Original",  "GhostNet_RecPlot"),
        ("MobileNet_Original",  "ConvNeXtV2_Original"),
        ("MobileNet_RecPlot",   "EffNet_Original"),
        ("MobileNet_RecPlot",   "EffNet_RecPlot"),
        ("MobileNet_RecPlot",   "GhostNet_Original"),
        ("MobileNet_RecPlot",   "GhostNet_RecPlot"),
        ("MobileNet_RecPlot",   "ConvNeXtV2_RecPlot"),
        ("EffNet_Original",     "EffNet_RecPlot"),
        ("EffNet_Original",     "GhostNet_Original"),
        ("EffNet_Original",     "GhostNet_RecPlot"),
        ("EffNet_Original",     "ConvNeXtV2_Original"),
        ("EffNet_RecPlot",      "GhostNet_Original"),
        ("EffNet_RecPlot",      "GhostNet_RecPlot"),
        ("EffNet_RecPlot",      "ConvNeXtV2_RecPlot"),
        ("GhostNet_Original",   "GhostNet_RecPlot"),
        ("GhostNet_Original",   "ConvNeXtV2_Original"),
        ("GhostNet_RecPlot",    "ConvNeXtV2_RecPlot"),
        ("ConvNeXtV2_Original", "ConvNeXtV2_RecPlot"),
    ]

    for seed in seeds:
        # Carrega os modelos disponíveis da seed atual usando o metadado
        modelos_ativos = {}
        for nome_amigavel, meta in model_meta.items():
            modelo = results[seed].get(meta["key"])
            if modelo is not None:
                modelo.eval()
                modelos_ativos[nome_amigavel] = modelo

        if not modelos_ativos:
            print(f"[SKIP seed {seed}] Nenhum modelo disponível, pulando.")
            continue

        print(f"\n[seed {seed}] Avaliando {len(modelos_ativos)} redes individuais...")

        y_true = []
        
        # Dicionários para acumular as predições de cada cenário
        preds_cenarios = {nome: [] for nome in modelos_ativos.keys()}
        
        # Filtra os ensembles onde AMBOS os modelos estão carregados nesta seed
        ensembles_ativos = [e for e in ensembles_config if e[0] in modelos_ativos and e[1] in modelos_ativos]
        for ens in ensembles_ativos:
            nome_ens = f"{ens[0]} + {ens[1]}"
            preds_cenarios[nome_ens] = []

        # Loop de inferência (Forward único por modelo!)
        with torch.no_grad():
            for imgs_orig, imgs_rec, labels in test_loader:
                imgs_orig = imgs_orig.to(DEVICE)
                imgs_rec  = imgs_rec.to(DEVICE)
                y_true.extend(labels.cpu().numpy())

                # Executa o forward APENAS UMA VEZ para cada modelo ativo
                outputs_softmax = {}
                for nome_modelo, model in modelos_ativos.items():
                    img = imgs_orig if model_meta[nome_modelo]["input"] == "orig" else imgs_rec
                    outputs_softmax[nome_modelo] = F.softmax(model(img), dim=1)
                    
                    # Salva predição da rede solo
                    y_pred_solo = torch.argmax(outputs_softmax[nome_modelo], dim=1).cpu().numpy()
                    preds_cenarios[nome_modelo].extend(y_pred_solo)

                # Calcula os Ensembles aproveitando os tensores já computados
                for ens in ensembles_ativos:
                    nome_ens = f"{ens[0]} + {ens[1]}"
                    soma_probabilidades = outputs_softmax[ens[0]] + outputs_softmax[ens[1]]
                    y_pred_ens = torch.argmax(soma_probabilidades, dim=1).cpu().numpy()
                    preds_cenarios[nome_ens].extend(y_pred_ens)

        # Calcula as métricas e salva os plots para tudo
        for cenario, y_pred in preds_cenarios.items():
            metrics = mostrar_metricas(nome_cenario=cenario, y_true=y_true, y_pred=y_pred)
            metrics["seed"] = seed
            metrics["cenario"] = cenario
            
            # Adiciona uma coluna identificadora para facilitar o agrupamento depois!
            metrics["tipo_scen"] = "Individual" if " + " not in cenario else "Ensemble"
            
            all_results.append(metrics)

            fname = f"seed{seed}__{cenario.replace(' + ', '__')}.png".replace(" ", "_")
            salvar_matriz_confusao(y_true, y_pred, class_names, save_path=os.path.join(plots_dir, fname))

    if not all_results:
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados brutos de teste salvos em {csv_path}")
    return df_results