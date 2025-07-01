# Classificação de Segmentos de Imagem - UCI Dataset

Projeto de classificação de regiões de imagem usando o dataset Image Segmentation da UCI. Técnicas aplicadas para melhorar a acurácia incluem pré-processamento, imputação de valores ausentes, normalização, seleção de atributos com LinearSVC (penalty L1) e uso de SVM para classificação. O desempenho foi comparado entre dados brutos e tratados.

## Dataset

- Origem: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/50/image+segmentation)
- Instâncias: 2310 (210 treino + 2100 teste)
- Atributos: 19 contínuos
- Classes: brickface, sky, foliage, cement, window, path, grass

## Técnicas Aplicadas

- Remoção de ruídos e registros inválidos
- Conversão de zeros para NaN e imputação com média
- Detecção de baixa representação por atributo
- Remoção de atributos irrelevantes: `region-pixel-count`, `short-line-density-5`
- Seleção de atributos com `SelectFromModel` e `LinearSVC`
- Normalização com `StandardScaler`
- Treinamento com `SVC` (kernel RBF)
- Avaliação com matriz de confusão e classification report

## Execução

1. Instale as dependências:

```bash
pip install pandas numpy scikit-learn
```

2. Execute:

```bash
python main.py
```

## Estrutura

```
├── segmentation.data     # Arquivo de dados
├── main.py               # Código do projeto
```

## Resultados

Modelos treinados com dados tratados apresentaram métricas superiores (precisão, recall, f1-score) em comparação aos dados brutos, evidenciando o impacto positivo do pipeline de pré-processamento.

## Autora

Luana – Desenvolvedora em IA aplicada e otimização de modelos.

## Referência

Image Segmentation [Dataset]. (1990). UCI Machine Learning Repository. https://doi.org/10.24432/C5GP4N

