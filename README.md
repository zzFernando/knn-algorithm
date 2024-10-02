# Algoritmo K-Nearest Neighbors (KNN)

Este repositório contém a implementação do algoritmo **K-Nearest Neighbors (KNN)**. O principal objetivo desta atividade prática é compreender o processo de classificação utilizando o KNN, explorando variações no valor de K, normalização de dados e a influência da dimensionalidade dos atributos.

## Objetivos da Atividade

- Compreender o processo de classificação com o KNN.
- Avaliar o impacto de diferentes valores de K na acurácia do modelo.
- Analisar os efeitos da normalização de dados no desempenho do KNN.
- Explorar como a dimensionalidade dos atributos afeta o modelo.
- Investigar o impacto de perturbações em atributos individuais nos resultados do modelo.

## Estrutura dos Dados

Os dados utilizados referem-se à classificação binária da qualidade de vinhos tintos (alta qualidade: `class=1`; baixa qualidade: `class=0`). O conjunto de dados foi pré-processado e está dividido da seguinte forma:

- **Conjunto de Treinamento**: 44 instâncias para treinar o modelo.
- **Conjunto de Teste**: 4 instâncias para avaliar o desempenho do modelo.

### Organização dos Dados

Os dados estão organizados em quatro diretórios principais:

- **Dados_Originais_2Features**: Dados não normalizados com dois atributos (`dióxido de enxofre total` e `ácido cítrico`).
- **Dados_Normalizados_2Features**: Dados normalizados com Min-Max, mantendo os dois mesmos atributos.
- **Dados_Originais_11Features**: Dados sem normalização com os 11 atributos originais.
- **Dados_Normalizados_11Features**: Dados normalizados com Min-Max, com os 11 atributos.

## Como Usar

### Requisitos

- **Python 3.12.5** ou superior.
- Bibliotecas Python necessárias:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Instale as dependências com o seguinte comando:

```bash
pip install -r requirements.txt
```

### Executando os Experimentos

Os experimentos estão divididos em três partes principais:

1. **Experimento A: Variação de K e Normalização**
   
   Este experimento avalia a variação de K (1, 3, 5, 7) no KNN, utilizando dados normalizados e não normalizados com dois atributos (`dióxido de enxofre total` e `ácido cítrico`). O objetivo é comparar a acurácia em diferentes cenários e analisar o impacto da normalização.

   Para rodar o Experimento A, execute:

   ```bash
   python ab.py
   ```

2. **Experimento B: Análise dos Vizinhos Mais Próximos**
   
   Este experimento foca em identificar os vizinhos mais próximos da instância de teste N1 para um modelo treinado com K=5 e dados não normalizados. O objetivo é visualizar a disposição desses vizinhos no espaço.

   Para rodar o Experimento B, execute:

   ```bash
   python ab.py
   ```

3. **Experimento C: Perturbação nos Atributos**
   
   Neste experimento, o atributo `ácido cítrico` da instância de teste N4 é perturbado, e o impacto é avaliado para dois modelos (um com 2 atributos e outro com 11 atributos). A ideia é verificar como a alteração de um atributo afeta a classificação do KNN.

   Para rodar o Experimento C, execute:

   ```bash
   python c.py
   ```

### Utilizando o Arquivo `knn.py`

O arquivo `knn.py` contém a lógica para treinar os modelos KNN e gerar gráficos baseados nos datasets fornecidos. Suas funcionalidades incluem:

- **Treinamento dos modelos**: Treina o KNN para diferentes valores de K e datasets (2 ou 11 atributos, normalizados ou não).
  
- **Geração de gráficos**: Após o treinamento, o script cria gráficos que mostram os K vizinhos mais próximos das instâncias de teste, salvando-os automaticamente na pasta `results/`.

Para usar o `knn.py`, basta rodar o seguinte comando:

```bash
python knn.py
```

### Parâmetros Personalizáveis

- **Valor de K**: Pode ser ajustado diretamente no código.
- **Normalização**: Experimentos com e sem normalização podem ser realizados.
- **Perturbação de Atributos**: No experimento C, o valor do atributo `ácido cítrico` pode ser alterado no código para analisar o impacto na classificação.

## Visualizando os Resultados

Os resultados de cada experimento são salvos na pasta `results/` como gráficos. Esses gráficos ilustram os vizinhos mais próximos de cada instância de teste e suas respectivas distâncias.

## Ferramentas Utilizadas

- **Python e scikit-learn**: Para a implementação do KNN e manipulação dos dados.
- **Pandas e Numpy**: Para manipulação eficiente dos dados.
- **Matplotlib**: Para visualização dos vizinhos mais próximos e geração de gráficos.

## Resultados

Os principais resultados dos experimentos incluem:

1. **Acurácia** do modelo KNN para diferentes valores de K.
2. **Visualização dos vizinhos mais próximos** para as instâncias de teste.
3. **Impacto da normalização** e de perturbações em atributos no desempenho do modelo.
