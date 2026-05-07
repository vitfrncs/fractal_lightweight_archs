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
    """Avalia cenários dos ensembles e salva métricas + matrizes"""
    all_results = []

    cenarios = [
        ("MobileNet_Original",  "MobileNet_RecPlot"),
        ("MobileNet_Original",  "EffNet_Original"),
        ("MobileNet_Original",  "EffNet_RecPlot"),
        ("MobileNet_Original",  "GhostNet_Original"),
        ("MobileNet_Original",  "GhostNet_RecPlot"),
        ("MobileNet_RecPlot",   "EffNet_Original"),
        ("MobileNet_RecPlot",   "EffNet_RecPlot"),
        ("MobileNet_RecPlot",   "GhostNet_Original"),
        ("MobileNet_RecPlot",   "GhostNet_RecPlot"),
        ("EffNet_Original",     "EffNet_RecPlot"),
        ("EffNet_Original",     "GhostNet_Original"),
        ("EffNet_Original",     "GhostNet_RecPlot"),
        ("EffNet_RecPlot",      "GhostNet_Original"),
        ("EffNet_RecPlot",      "GhostNet_RecPlot"),
        ("GhostNet_Original",   "GhostNet_RecPlot"),
        # modelos sozinhos
        ("MobileNet_Original",  "MobileNet_Original"),
        ("MobileNet_RecPlot",   "MobileNet_RecPlot"),
        ("EffNet_Original",     "EffNet_Original"),
        ("EffNet_RecPlot",      "EffNet_RecPlot"),
        ("GhostNet_Original",   "GhostNet_Original"),
        ("GhostNet_RecPlot",    "GhostNet_RecPlot"),
    ]

    model_input = {
        "MobileNet_Original":  "orig",
        "MobileNet_RecPlot":   "rec",
        "EffNet_Original":     "orig",
        "EffNet_RecPlot":      "rec",
        "GhostNet_Original":   "orig",
        "GhostNet_RecPlot":    "rec",
    }

    for seed in seeds:
        modelos = {
            "MobileNet_Original":  results[seed].get('mobnet_orig'),
            "MobileNet_RecPlot":   results[seed].get('mobnet_recplot'),
            "EffNet_Original":     results[seed].get('effnet_orig'),
            "EffNet_RecPlot":      results[seed].get('effnet_recplot'),
            "GhostNet_Original":   results[seed].get('ghostnet_orig'),
            "GhostNet_RecPlot":    results[seed].get('ghostnet_recplot'),
        }

        # Identifica quais modelos estão disponíveis (não-None) para esta seed
        modelos_disponiveis = {k for k, v in modelos.items() if v is not None}
        if not modelos_disponiveis:
            print(f"[SKIP seed {seed}] Nenhum modelo disponível, pulando.")
            continue

        # Filtra apenas cenários onde AMBOS os modelos estão disponíveis
        cenarios_ativos = [
            c for c in cenarios
            if c[0] in modelos_disponiveis and c[1] in modelos_disponiveis
        ]

        if not cenarios_ativos:
            print(f"[SKIP seed {seed}] Nenhum cenário válido com os modelos disponíveis.")
            continue

        print(f"\n[seed {seed}] Modelos disponíveis: {sorted(modelos_disponiveis)}")
        print(f"[seed {seed}] Cenários ativos: {len(cenarios_ativos)}")

        for model in modelos.values():
            if model is not None:
                model.eval()

        y_true = []
        preds_cenarios = {c: [] for c in cenarios_ativos}

        with torch.no_grad():
            for imgs_orig, imgs_rec, labels in test_loader:
                imgs_orig = imgs_orig.to(DEVICE)
                imgs_rec  = imgs_rec.to(DEVICE)

                # Forward apenas dos modelos disponíveis
                outputs = {}
                for name, model in modelos.items():
                    if model is not None:
                        img = imgs_orig if model_input[name] == "orig" else imgs_rec
                        outputs[name] = F.softmax(model(img), dim=1)

                y_true.extend(labels.cpu().numpy())

                for cenario in cenarios_ativos:
                    # Ambos já foram validados acima, mas checagem defensiva
                    if cenario[0] not in outputs or cenario[1] not in outputs:
                        continue
                    soma = outputs[cenario[0]] + outputs[cenario[1]]
                    y_pred_batch = torch.argmax(soma, dim=1).cpu().numpy()
                    preds_cenarios[cenario].extend(y_pred_batch)

        # Calcula métricas — só para cenários com predições completas
        for cenario, y_pred in preds_cenarios.items():
            if len(y_pred) != len(y_true):
                print(f"[WARN] Cenário {cenario} com predições incompletas "
                      f"({len(y_pred)} vs {len(y_true)}), pulando.")
                continue

            metrics = mostrar_metricas(
                nome_cenario=" + ".join(cenario),
                y_true=y_true,
                y_pred=y_pred,
            )
            metrics["seed"] = seed
            metrics["cenario"] = " + ".join(cenario)
            all_results.append(metrics)

            fname = f"seed{seed}__{cenario[0]}__{cenario[1]}.png".replace(" ", "_")
            salvar_matriz_confusao(
                y_true, y_pred, class_names,
                save_path=os.path.join(plots_dir, fname),
            )

    if not all_results:
        print("Nenhum resultado calculado. Verifique se os modelos foram carregados.")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em {csv_path}")
    return df_results