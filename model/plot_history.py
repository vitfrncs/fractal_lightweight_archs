import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO DE CAMINHOS
# =========================================================
DATASET_NOME = "displasia"
SEED_REF = 42  
BASE_DIR = f"outputs/results_kfold/{DATASET_NOME}/{SEED_REF}"
OUTPUT_DIR = f"outputs/{DATASET_NOME}/plots_treino"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeamento dos cenários individuais para as pastas correspondentes
CENARIOS_INDIVIDUAIS = [
    {"nome": "MobileNet_Original",  "pasta": "originais", "backbone": "mobilenet"},
    {"nome": "MobileNet_RecPlot",   "pasta": "F-RecPlot", "backbone": "mobilenet"},
    {"nome": "EffNet_Original",     "pasta": "originais", "backbone": "efficientnet_b0"},
    {"nome": "EffNet_RecPlot",      "pasta": "F-RecPlot", "backbone": "efficientnet_b0"},
    {"nome": "GhostNet_Original",   "pasta": "originais", "backbone": "ghostnet"},
    {"nome": "GhostNet_RecPlot",    "pasta": "F-RecPlot", "backbone": "ghostnet"},
    {"nome": "ConvNeXtV2_Original", "pasta": "originais", "backbone": "convnextv2"},
    {"nome": "ConvNeXtV2_RecPlot",  "pasta": "F-RecPlot", "backbone": "convnextv2"},
]

def extrair_lista(string_dados):
    """Converte a string do CSV de volta para uma lista Python de floats."""
    return ast.literal_eval(string_dados)

def plotar_curvas_cenario(cenario):
    path_history = os.path.join(
        BASE_DIR, cenario["pasta"], cenario["backbone"], "history.csv"
    )
    
    if not os.path.exists(path_history):
        print(f"[SKIP] Histórico não encontrado para {cenario['nome']}")
        return

    df = pd.read_csv(path_history)
    
    # Listas para acumular as curvas de todos os folds do K-Fold
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    
    for _, row in df.iterrows():
        all_train_loss.append(extrair_lista(row["train_loss"]))
        all_val_loss.append(extrair_lista(row["val_loss"]))
        all_train_acc.append(extrair_lista(row["train_accuracy"]))
        all_val_acc.append(extrair_lista(row["val_accuracy"]))
        
    # Transforma em arrays numpy para calcular a média da evolução entre os folds
    mean_train_loss = np.mean(all_train_loss, axis=0)
    mean_val_loss = np.mean(all_val_loss, axis=0)
    mean_train_acc = np.mean(all_train_acc, axis=0)
    mean_val_acc = np.mean(all_val_acc, axis=0)
    
    epocas = range(1, len(mean_train_loss) + 1)
    
    # Criar a figura com subplots lado a lado (Loss e Acurácia)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Histórico de Treinamento (Média dos Folds) - {cenario['nome']}", fontsize=14, fontweight='bold')
    
    # 1. Plot de Loss
    ax1.plot(epocas, mean_train_loss, label="Treino", color="royalblue", linewidth=2)
    ax1.plot(epocas, mean_val_loss, label="Validação", color="darkorange", linewidth=2, linestyle="--")
    ax1.set_title("Evolução do Loss (Função de Custo)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # 2. Plot de Acurácia
    ax2.plot(epocas, mean_train_acc, label="Treino", color="royalblue", linewidth=2)
    ax2.plot(epocas, mean_val_acc, label="Validação", color="darkorange", linewidth=2, linestyle="--")
    ax2.set_title("Evolução da Acurácia")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Acurácia")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    # Salvar o gráfico gerado
    save_path = os.path.join(OUTPUT_DIR, f"curva_{cenario['nome']}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico salvo para {cenario['nome']} em: {save_path}")

# Executa para todas as redes individuais
if __name__ == "__main__":
    print("Iniciando geração dos gráficos de treinamento...")
    for cenario in CENARIOS_INDIVIDUAIS:
        plotar_curvas_cenario(cenario)