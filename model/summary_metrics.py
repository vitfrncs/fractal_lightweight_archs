import pandas as pd
import os

CSV_PATH = "outputs/displasia/resultados_testes.csv"
METRICAS = ["acc", "f1_macro", "recall_macro", "specificity_macro"]

CSV_SAIDA_INDIVIDUAIS = "outputs/displasia/resumo_metricas_redes_individuais.csv"
CSV_SAIDA_ENSEMBLES   = "outputs/displasia/resumo_metricas_ensembles.csv"

# 1. Carregar CSV gerado pelo metrics.py
df = pd.read_csv(CSV_PATH)

def gerar_tabela_resumo(dataframe_filtrado):
    """Agrupa por cenário, calcula Média ± DP e ordena pelo F1-Score."""
    grouped = dataframe_filtrado.groupby("cenario")[METRICAS]
    mean_df = grouped.mean()
    std_df = grouped.std()

    tabela = pd.DataFrame(index=mean_df.index)
    for metrica in METRICAS:
        tabela[metrica] = (
            mean_df[metrica].round(4).astype(str) + " ± " + std_df[metrica].round(4).astype(str)
        )

    tabela["_sort"] = mean_df["f1_macro"]
    tabela = tabela.sort_values("_sort", ascending=False).drop(columns="_sort")
    return tabela

# 2. Separar usando a nova coluna de controle estruturado
df_indiv = df[df["tipo_scen"] == "Individual"]
df_ensem = df[df["tipo_scen"] == "Ensemble"]

# 3. Gerar os dataframes finais formatados
tabela_individuais = gerar_tabela_resumo(df_indiv)
tabela_ensembles   = gerar_tabela_resumo(df_ensem)

# 4. Salvar os relatórios em escopos bem definidos
tabela_individuais.to_csv(CSV_SAIDA_INDIVIDUAIS)
tabela_ensembles.to_csv(CSV_SAIDA_ENSEMBLES)

print("=== TABELA DE REDES INDIVIDUAIS NO TESTE ===")
print(tabela_individuais)
print(f"\nSalvo em: {CSV_SAIDA_INDIVIDUAIS}\n")

print("=== TABELA DE ENSEMBLES NO TESTE ===")
print(tabela_ensembles)
print(f"Salvo em: {CSV_SAIDA_ENSEMBLES}")