# Algoritmo KNN

Este repositório contém a implementação do algoritmo K-Nearest Neighbors (KNN). O objetivo principal é experimentar o KNN variando o valor de k, aplicando normalização nos dados e analisando os efeitos de diferentes dimensionalidades de atributos.

## Objetivo da Atividade

- Compreender o processo de classificação utilizando o KNN.
- Avaliar o impacto de diferentes valores de k.
- Analisar os efeitos da normalização dos dados e da dimensionalidade dos atributos.
- Testar o impacto de perturbações nos atributos sobre os resultados do modelo.

## Como Usar

### Requisitos

- **Python 3.12.5** ou superior.
- Bibliotecas Python necessárias:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Você pode instalar as dependências com o comando:

```bash
pip install -r requirements.txt
```

### Executando os Experimentos

O repositório contém três principais scripts que executam diferentes experimentos com o algoritmo KNN. Veja abaixo como executá-los:

1. **Experimento A: Variação de K e Normalização**
   
   Este experimento treina o modelo KNN com diferentes valores de K (1, 3, 5, 7) usando os dados originais e normalizados com 2 atributos. Ele compara a acurácia e visualiza os vizinhos mais próximos para instâncias de teste.

   Para rodar o experimento A, use:

   ```bash
   python ab.py
   ```

2. **Experimento B: Análise dos Vizinhos Mais Próximos**

   O experimento B foca em visualizar os vizinhos mais próximos para a instância N1 do conjunto de teste e estender essa análise para N2, N3 e N4, utilizando K = 5.

   Para rodar o experimento B, use:

   ```bash
   python ab.py
   ```

3. **Experimento C: Perturbações no Atributo**

   Neste experimento, dois modelos KNN são treinados (um com 2 atributos e outro com 11 atributos). A instância de teste N4 é perturbada alterando o valor do atributo "citric acid", e o efeito dessas perturbações é analisado em relação aos vizinhos mais próximos.

   Para rodar o experimento C, use:

   ```bash
   python c.py
   ```

### Visualizando os Resultados

Cada experimento gera gráficos que mostram os vizinhos mais próximos para as instâncias de teste, com base nas distâncias euclidianas calculadas pelo KNN. As imagens dos resultados são salvas na pasta `results/`.

### Parâmetros Personalizáveis

- **K**: Valor de K pode ser modificado diretamente no código para avaliar diferentes configurações.
- **Normalização**: Há conjuntos de dados normalizados e não normalizados para analisar os impactos dessa técnica no desempenho do modelo.
- **Perturbação de Atributos**: No experimento C, o valor do atributo "citric acid" pode ser alterado diretamente no código para observar os impactos na classificação.

Boa prática de KNN!