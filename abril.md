# O que mudou no projeto

Esse documento compara a versão original dos arquivos com a versão atual, explicando o raciocínio por trás de cada mudança.

---

## Estrutura de pastas

A mudança mais visível foi na organização geral. Antes tudo ficava numa pasta só:

```
model/
├── utils.py
├── dataset.py
├── model.py
├── metrics.py
├── train.py
└── run.py
```

Agora tem uma pasta `architectures/` separada ao lado de `model/`:

```
projeto/
├── architectures/
│   ├── __init__.py
│   ├── mobilenet.py
│   ├── efficientnet.py
│   └── ghostnet.py
├── model/
│   ├── utils.py
│   ├── dataset.py
│   ├── model.py        ← virou uma factory
│   ├── metrics.py
│   ├── train.py
│   ├── run.py
│   ├── save_outputs.py ← novo
│   ├── gradcam.py      ← novo
│   └── efficiency.py   ← novo
└── outputs/
    ├── models/{seed}/
    ├── results_kfold/
    ├── logits/
    ├── gradcam/
    └── plots/
```

A motivação foi simples: o projeto vai crescer com mais arquiteturas, e misturar a definição das redes com o código de treino ia ficar bagunçado. Agora cada arquivo em `architectures/` define uma rede e só isso — sem depender de nada do restante do projeto.

---

## `model.py` — virou uma factory

**Antes**, o `model.py` tinha as funções `criar_modelo` e `carregar_modelo` com `if/elif` hardcoded para MobileNet e EfficientNet:

```python
# antes
if backbone == "mobilenet":
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(...)
elif backbone == "efficientnet_b0":
    model = models.efficientnet_b0(pretrained=pretrained)
    ...
```

**Agora** tem um dicionário `REGISTRO` que mapeia o nome do backbone para a função construtora. Para adicionar uma nova arquitetura, você só acrescenta uma linha no registro — o resto do código não muda:

```python
# agora
REGISTRO = {
    "mobilenet":       criar_mobilenet,
    "efficientnet_b0": criar_efficientnet_b0,
    "ghostnet":        criar_ghostnet,
}
```

A lógica de `criar_modelo` e `carregar_modelo` em si não mudou, só onde as arquiteturas são definidas.

---

## `train.py` — três adições

### 1. Medição de tempo de treino

O `train_one_fold` ganhou um timer simples ao redor do loop de épocas. O tempo total fica salvo em `metrics["train_time_seconds"]` e vai para o CSV de resultados por fold. Útil para a análise de eficiência computacional do artigo.

```python
start_time = time.time()
# ... loop de épocas ...
metrics["train_time_seconds"] = time.time() - start_time
```

### 2. `BACKBONE_REGISTRO` e controle de quais backbones treinar

Antes o `train_seeds` tinha os backbones hardcoded dentro do loop. Agora tem uma lista `BACKBONE_REGISTRO` no topo do arquivo, e `train_seeds` aceita um parâmetro `backbones` para você escolher o que treinar:

```python
# treinar só a GhostNet
train_seeds(SEEDS, dataset_path, classes, backbones=['ghostnet'])

# treinar tudo
train_seeds(SEEDS, dataset_path, classes)
```

### 3. `skip_existing`

Se o `.pth` de um modelo já existe em `outputs/models/{seed}/`, o treino pula aquela combinação automaticamente. Isso evita retreinar o que já foi feito se o script cair no meio ou se você quiser só adicionar uma arquitetura nova.

### 4. Correção de caminho (bug)

O `run_kfold` original salvava os pesos em `models/{seed}/` (relativo à pasta `model/`), mas o `skip_existing` verificava em `../outputs/models/{seed}/`. Os caminhos não batiam, então o skip nunca funcionava. Agora ambos apontam para `../outputs/models/{seed}/`.

Os CSVs de resultado por fold também foram movidos de `results_kfold/` para `../outputs/results_kfold/`, mantendo tudo dentro de `outputs/`.

---

## `metrics.py` — duas adições e um bug

### 1. `salvar_matriz_confusao`

Função nova que gera e salva a matriz de confusão como PNG usando seaborn. É chamada dentro de `metrics_to_csv` para cada cenário avaliado, salvando em `outputs/plots/`.

### 2. GhostNet nos cenários

Os cenários de ensemble foram expandidos para incluir a GhostNet. Com 3 arquiteturas × 2 tipos de entrada são 21 cenários no total (15 pares + 6 baselines individuais).

### 3. Guards para modelos `None` e correção de bug no CSV

Quando `skip_existing=True`, alguns modelos ficam como `None` no dicionário `results` até serem carregados. O código original chamava `model.eval()` sem checar, o que quebrava com `AttributeError`. Agora todos os loops verificam antes de usar o modelo.

Também havia um bug no `to_csv`: estava passando a string literal `"csv_path"` em vez da variável `csv_path`, então o arquivo era sempre salvo com o nome errado. Corrigido.

---

## Arquivos novos

### `save_outputs.py`

Salva logits, probabilidades e rótulos de todas as amostras em arquivos `.npy`, separados por seed, backbone, tipo de entrada e split (val/test). Isso é importante para poder analisar as predições depois sem precisar rodar a inferência de novo, e também é pedido explicitamente como parte dos experimentos.

Estrutura de saída:
```
outputs/logits/{seed}/{backbone}_{dataset_type}_{split}/
    logits.npy   — shape (N, num_classes)
    probs.npy    — shape (N, num_classes)
    labels.npy   — shape (N,)
```

### `gradcam.py`

Gera mapas Grad-CAM para visualizar quais regiões da imagem cada modelo está usando para classificar. Funciona em lote: você passa um DataLoader e ele gera N imagens por classe, salvando em `outputs/gradcam/{backbone}/{split}/{classe}/`.

A camada alvo para o Grad-CAM é selecionada automaticamente pelo nome do backbone. Para adicionar um novo backbone, basta incluir um `elif` na função `_get_target_layer` com a camada correta — o guia `adicionar_arquitetura.md` explica como descobrir qual camada usar.

### `efficiency.py`

Mede três coisas para cada modelo: número de parâmetros treináveis (em milhões), FLOPs aproximados via `thop`, e latência média de inferência por imagem em milissegundos. Salva tudo em `outputs/eficiencia.csv`, que serve como base para a tabela de eficiência computacional do artigo.

---

## `run.py` — virou o orquestrador

O `run.py` original só fazia treino e avaliação. Agora ele executa o pipeline completo em sequência:

1. Treina os backbones selecionados
2. Carrega os melhores pesos de todos os modelos
3. Avalia os cenários de ensemble → CSV + matrizes de confusão
4. Salva logits e probabilidades (val + teste)
5. Gera mapas Grad-CAM
6. Mede e salva a eficiência computacional

A linha de treino mudou para refletir que MobileNet e EfficientNet já foram treinados:

```python
# antes
results = train_seeds(SEEDS, dataset_path, classes)

# agora
results = train_seeds(SEEDS, dataset_path, classes,
                      backbones=['ghostnet'], skip_existing=True)
```

---

## `utils.py` e `dataset.py`

Não mudaram. O `utils.py` continua com as constantes globais (device, seeds, transforms) e o `dataset.py` com `ImageDataset` e `EnsembleTestDataset` exatamente como antes.