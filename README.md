# Fractal Features and Lightweight Architectures

Repositório de desenvolvimento de um pipeline para extração de características fractais de imagens, geração de representações visuais dessas características e experimentação com modelos de aprendizado de máquina.

O projeto reúne ferramentas de processamento de imagens, organização de dados e treinamento e avaliação de modelos, com foco na integração entre descritores de imagem e arquiteturas de redes neurais.

## Objetivo

Organizar as etapas de processamento de imagens e experimentação com modelos em um fluxo de trabalho que permita extrair características, gerar representações para análise e utilizar os resultados em experimentos de aprendizado de máquina.

## Pipeline do projeto

O repositório está organizado em módulos que contemplam diferentes etapas do processamento.

### 1. Extração de características fractais

O diretório `extract_fractal_features/` contém scripts e utilitários para calcular descritores fractais e organizar os resultados.

Entre as medidas mencionadas na implementação estão PMR, lacunaridade, D, N e clustperc.

Os resultados podem ser exportados para arquivos CSV, permitindo sua utilização nas etapas seguintes.

### 2. Transformação das características em representações visuais

O diretório `feature_to_image/` reúne ferramentas para gerar imagens a partir das características extraídas.

O pipeline principal integra essa etapa ao processamento dos dados, permitindo produzir arquivos tabulares e representações visuais organizadas na estrutura de saída.

### 3. Arquiteturas de redes neurais

O diretório `architectures/` organiza as implementações das arquiteturas utilizadas no projeto.

Essa separação permite manter as definições dos modelos independentes de outras partes do fluxo experimental e facilita a inclusão de novas arquiteturas.

O arquivo `add_architectures.md` documenta o procedimento utilizado para adicionar um novo backbone à estrutura existente.

### 4. Treinamento e avaliação

O diretório `model/` reúne módulos relacionados à preparação dos conjuntos de dados, definição dos modelos, treinamento e execução das avaliações.

Essa organização separa as etapas de modelagem e permite estruturar os experimentos em módulos específicos.

### 5. Pipeline de processamento

O arquivo `main_pipeline.py` coordena o processamento das imagens e a geração dos arquivos de saída.

De acordo com sua implementação, o pipeline realiza as seguintes etapas:

1. Percorre os diretórios de entrada em busca de imagens.
2. Executa a extração de características.
3. Organiza os resultados em estruturas tabulares.
4. Exporta arquivos CSV com os descritores e informações associadas às imagens.
5. Executa a geração de representações visuais a partir das características.
6. Organiza os resultados no diretório de saída.

## Estrutura do repositório

```text
fractal_lightweight_archs/
├── architectures/
├── extract_fractal_features/
├── feature_to_image/
├── model/
├── validacoes/
├── main_pipeline.py
├── add_architectures.md
├── requirements.txt
└── README.md
```

Os diretórios e arquivos auxiliares contêm as implementações específicas de cada etapa. A estrutura acima apresenta apenas os principais componentes do repositório.

## Tecnologias

O projeto utiliza Python e bibliotecas para processamento de dados, manipulação de imagens e aprendizado de máquina.

As dependências estão especificadas em `requirements.txt`.

## Como utilizar

O pipeline principal recebe um diretório de entrada com imagens, um diretório de destino para os resultados e, opcionalmente, o modo de geração das imagens.

A interface de linha de comando está documentada no próprio `main_pipeline.py`.

Formato de execução:

```bash
python main_pipeline.py <diretorio_entrada> <diretorio_saida> [modo]
```

O parâmetro `modo` aceita as opções `both`, `recplot` ou `classical`, conforme indicado na implementação.

Antes de executar o pipeline, verifique as dependências, a organização do conjunto de dados e os requisitos dos scripts de processamento.

## Documentação adicional

- `main_pipeline.py`: coordenação do processamento e geração dos arquivos de saída.
- `add_architectures.md`: orientações para adicionar arquiteturas ao projeto.
- `requirements.txt`: dependências do ambiente Python.

## Créditos

Projeto vinculado ao LIPAI — Laboratório Interdisciplinar de Processamento e Análise de Imagens da UNIFESP.
