from model import criar_modelo

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import os
import pandas as pd
import time
import copy
import statistics

from dataset import load_data_from_folders, ImageDataset
from sklearn.model_selection import StratifiedKFold
from utils import *
from torch.utils.data import DataLoader

EPOCHS = 50
SEED = 42

# Registro: backbone → chaves no dicionário results 
BACKBONE_REGISTRO = [
    ("mobilenet",       "mobnet_orig",     "mobnet_recplot"),
    ("efficientnet_b0", "effnet_orig",     "effnet_recplot"),
    ("ghostnet",        "ghostnet_orig",   "ghostnet_recplot"),
    ("convnextv2",      "convnext_orig",   "convnext_recplot"),
]


def train_one_fold(model, train_loader, val_loader, epochs=EPOCHS):
    """Treina um fold do K-Fold. Usado apenas para AVALIAR a configuração
    (backbone/dataset_type) — o modelo resultante não é salvo como modelo final."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # Histórico por época
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    metrics = {
        "best_epoch": -1,
        "best_val_loss": float("inf"),
        "best_val_accuracy": -1,
        "best_train_loss": float("inf"),
        "best_train_accuracy": -1,
        "train_time_seconds":  0.0,
    }

    best_model_state = None
    start_time = time.time()

    for epoch in range(epochs):
        # ================= TREINO =================
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, _ , labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        # ================= VALIDAÇÃO =================
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, _ , labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        #MELHOR ÉPOCA DENTRO DO FOLD =================
        # Aqui o uso de val_loss é o padrão (early stopping / checkpoint por época).
        if val_loss < metrics["best_val_loss"]:
            metrics["best_val_loss"] = val_loss
            metrics["best_epoch"] = epoch + 1
            metrics["best_val_accuracy"] = val_acc
            metrics["best_train_loss"] = train_loss
            metrics["best_train_accuracy"] = train_acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Época {epoch+1}/{epochs} | "
            f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}"
        )

    metrics["train_time_seconds"] = time.time() - start_time
    return model, best_model_state, history, metrics


def train_final_model(model, train_loader, epochs):
    """Treina o modelo FINAL usando 100% dos dados (train+val do K-Fold).

    Sem validação: não há critério de early stopping/checkpoint por época,
    já que não sobrou dado para validar. O número de épocas é decidido
    previamente por quem chama esta função — normalmente a média (ou mediana)
    do `best_epoch` observado nos folds do K-Fold, que serviu apenas para
    avaliar a configuração (backbone/dataset_type)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, _, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        print(
            f"[FINAL] Época {epoch+1}/{epochs} | "
            f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f}"
        )

    train_time_seconds = time.time() - start_time
    return model, history, train_time_seconds


# K-Fold para um backbone/dataset_type
def run_kfold(dataset_path, dataset_type, class_names, dataset_nome, backbone="mobilenet", seed=SEED):
    """Executa K-Fold para AVALIAR a configuração (backbone/dataset_type) e,
    em seguida, retreina um único modelo final usando 100% dos dados
    (train+val), que é o modelo efetivamente salvo em disco e usado no teste.

    Ele serve apenas para:
      1) estimar o desempenho esperado da configuração (df_metrics/df_history);
      2) decidir por quantas épocas treinar o modelo final (média dos
         best_epoch observados nos folds).
    """
    os.makedirs(f"models/{seed}", exist_ok=True)

    data_list = load_data_from_folders(dataset_path, class_names, dataset_type)
    print(f"Total de imagens: {len(data_list)}")

    paths = [x[0] for x in data_list]
    labels = [x[1] for x in data_list]

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=seed
    )

    fold_results = []
    all_history = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print("\n============================")
        print(f"FOLD {fold+1}/{K_FOLDS}")
        print("============================")

        train_data = [(paths[i], labels[i]) for i in train_idx]
        val_data   = [(paths[i], labels[i]) for i in val_idx]

        train_dataset = ImageDataset(train_data, transform=transform)
        val_dataset   = ImageDataset(val_data, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        # modelo
        model = criar_modelo(
            backbone=backbone,
            num_classes=len(class_names),
            pretrained=True
        ).to(DEVICE)

        model, best_model_fold_state, history, metrics = train_one_fold(
            model, train_loader, val_loader
        )

        # histórico e resultados (só para avaliação da config) ----------
        fold_results.append({
            "fold": fold + 1,
            **metrics
        })

        all_history.append({
            "fold": fold + 1,
            **history
        })

        # NOTE: não salvamos mais nenhum .pth de fold individual aqui —
        # o modelo que vai para o teste é retreinado abaixo, com 100% dos dados.
        del model, best_model_fold_state
        torch.cuda.empty_cache()

    # ================= SALVAR RESULTADOS DO K-FOLD (avaliação da configuração) =================
    base_path = f"outputs/results_kfold/{dataset_nome}/{seed}/{dataset_type}/{backbone}"
    os.makedirs(base_path, exist_ok=True)

    # métricas finais por fold
    df_metrics = pd.DataFrame(fold_results)
    df_metrics.to_csv(f"{base_path}/metrics.csv", index=False)

    # histórico completo
    df_history = pd.DataFrame(all_history)
    df_history.to_csv(f"{base_path}/history.csv", index=False)

    print("\n===== RESULTADOS DO K-FOLD (avaliação da configuração) =====")
    print(df_metrics)
    print("\nMédias:")
    print(df_metrics.mean(numeric_only=True))

    # ================= DEFINIR Nº DE ÉPOCAS DO TREINO FINAL =================
    best_epochs = df_metrics["best_epoch"].tolist()
    epochs_final = round(statistics.mean(best_epochs))
    # Alternativa mais robusta a outliers:
    # epochs_final = round(statistics.median(best_epochs))

    print(
        f"\nÉpocas escolhidas para o treino final "
        f"(média dos best_epoch dos {K_FOLDS} folds): {epochs_final}"
    )

    # ================= RETREINAR COM 100% DOS DADOS (TRAIN + VAL) =================
    full_data = [(paths[i], labels[i]) for i in range(len(paths))]
    full_dataset = ImageDataset(full_data, transform=transform)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    final_model = criar_modelo(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=True
    ).to(DEVICE)

    final_model, final_history, final_train_time = train_final_model(
        final_model, full_loader, epochs=epochs_final
    )

    # histórico do treino final
    pd.DataFrame(final_history).to_csv(f"{base_path}/final_train_history.csv", index=False)

    # resumo do treino final
    final_summary = pd.DataFrame([{
        "epochs_final": epochs_final,
        "final_train_loss": final_history["train_loss"][-1],
        "final_train_accuracy": final_history["train_accuracy"][-1],
        "train_time_seconds": final_train_time,
    }])
    final_summary.to_csv(f"{base_path}/final_train_summary.csv", index=False)

    print("\n===== TREINO FINAL (100% dos dados) =====")
    print(final_summary)

    # ================= SALVAR MODELO FINAL (o que vai para o teste) =================
    out_dir = f"outputs/models/{dataset_nome}/{seed}"
    os.makedirs(out_dir, exist_ok=True)
    nome_modelo = f"{out_dir}/{backbone}_{dataset_type}.pth"
    torch.save(final_model.state_dict(), nome_modelo)
    print(f"   -> Modelo final (retreinado com 100% dos dados) salvo em {nome_modelo}")

    return final_model


def train_seeds(seeds, dataset_path, classes,  dataset_nome, backbones=None, skip_existing=True):
    """Treina backbones × dataset_types para cada seed.

    Para cada combinação (backbone, dataset_type, seed):
        1) roda K-Fold para avaliar a configuração (gera metrics.csv/history.csv);
        2) retreina um único modelo final com 100% dos dados (train+val);
        3) salva esse modelo final em outputs/models/... — é esse .pth que
           deve ser usado no teste.

    Mudanças:
        - backbones: Lista de backbones a treinar.
            Se = None, treina todos os registrados em BACKBONE_REGISTRO.
        - skip_existing: Se True, pula combinações cujo .pth do modelo final
            já existe em disco.

        Para treinar só uma rede, por exemplo: backbones=['ghostnet']
    """

    # filtra quais backbones rodar
    if backbones is not None:
        backbones_lower = [b.lower() for b in backbones] # lowercase
        registro_ativo = [
            entry for entry in BACKBONE_REGISTRO
            if entry[0] in backbones_lower
        ]
        nao_encontrados = set(backbones_lower) - {e[0] for e in BACKBONE_REGISTRO}
        if nao_encontrados:
            raise ValueError(
                f"Backbones não registrados: {nao_encontrados}. "
                f"Registre-os em BACKBONE_REGISTRY antes de treinar."
            )
    else:
        registro_ativo = BACKBONE_REGISTRO

    results = {}
    
    for seed in seeds:
        print(f"\n===== Treinando com seed {seed} =====")
        set_seed(seed)
        results[seed] = {}

        for backbone, chave_orig, chave_rec in registro_ativo:
 
            for dataset_type, chave in [("F-RecPlot", chave_rec),
                                         ("originais",  chave_orig)]:
 
                pth = f"outputs/models/{dataset_nome}/{seed}/{backbone}_{dataset_type}.pth"
 
                if skip_existing and os.path.exists(pth):
                    print(f"\n[SKIP] {backbone} / {dataset_type} / seed {seed} "
                          f"— modelo final já existe em {pth}")
                    results[seed][chave] = None   # será carregado depois no run.py
                    continue
 
                results[seed][chave] = run_kfold(
                    dataset_path, dataset_type, classes,
                    dataset_nome=dataset_nome,
                    backbone=backbone, seed=seed,
                )
                torch.cuda.empty_cache()
 
    return results
