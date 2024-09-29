import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Função para carregar os datasets com base na escolha do usuário
def load_dataset(choice):
    if choice == '1':
        train = pd.read_csv('Dados_Originais_2Features/TrainingData_2F_Original.txt', sep='\t')
        test = pd.read_csv('Dados_Originais_2Features/TestingData_2F_Original.txt', sep='\t')
    elif choice == '2':
        train = pd.read_csv('Dados_Normalizados_2Features/TrainingData_2F_Norm.txt', sep='\t')
        test = pd.read_csv('Dados_Normalizados_2Features/TestingData_2F_Norm.txt', sep='\t')
    elif choice == '3':
        train = pd.read_csv('Dados_Originais_11Features/TrainingData_11F_Original.txt', sep='\t')
        test = pd.read_csv('Dados_Originais_11Features/TestingData_11F_Original.txt', sep='\t')
    elif choice == '4':
        train = pd.read_csv('Dados_Normalizados_11Features/TrainingData_11F_Norm.txt', sep='\t')
        test = pd.read_csv('Dados_Normalizados_11Features/TestingData_11F_Norm.txt', sep='\t')
    else:
        print("Opção inválida.")
        return None, None
    return train, test

# Função para executar o KNN e calcular acurácia
def run_knn(X_train, y_train, X_test, y_test, k):
    # Criar e treinar o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = knn.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAcurácia do modelo com k={k}: {accuracy * 100:.2f}%')

    return knn

# Função para listar os vizinhos mais próximos
def list_nearest_neighbors(knn, X_test, instance_idx, k):
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    print(f"\nVizinhos mais próximos da instância {instance_idx}:")
    print(f"Distâncias: {distances}")
    print(f"Índices: {indices}")

# Função para modificar o atributo e ver o impacto
def perturb_and_classify(knn, X_test, original_instance, new_values, attribute):
    print("\nClassificação da instância perturbada:")
    for val in new_values:
        perturbed_instance = X_test[original_instance].copy()
        perturbed_instance[attribute] = val
        pred_class = knn.predict(perturbed_instance.reshape(1, -1))
        print(f"Valor {val} para {attribute}: Classe prevista: {pred_class[0]}")

# Função para visualizar os resultados
def plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx, k):
    # Plotar os dados de treinamento e teste
    plt.figure(figsize=(8, 6))

    # Plotar instâncias de treinamento
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Treinamento', cmap='coolwarm', edgecolor='k')

    # Plotar instâncias de teste
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', s=100, label='Teste', cmap='coolwarm', edgecolor='k')

    # Vizinhos mais próximos para a instância escolhida
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)

    # Plotar a instância de teste alvo
    plt.scatter(X_test[instance_idx, 0], X_test[instance_idx, 1], c='green', marker='x', s=200, label=f'Instância N{instance_idx+1}')

    # Conectar os vizinhos mais próximos
    for i in indices[0]:
        plt.plot([X_test[instance_idx, 0], X_train[i, 0]], [X_test[instance_idx, 1], X_train[i, 1]], 'k--', lw=1)

    # Legenda e títulos
    plt.title(f'Vizinhos mais próximos para a instância N{instance_idx+1} com k={k}')
    plt.xlabel('Total Sulfur Dioxide')
    plt.ylabel('Citric Acid')
    plt.legend()
    plt.show()

# Escolher dataset para treinamento e teste juntos (já com arquivos separados)
print("Escolha o dataset para Treinamento e Teste:")
print("1. Dados Originais (2 Features)")
print("2. Dados Normalizados (2 Features)")
print("3. Dados Originais (11 Features)")
print("4. Dados Normalizados (11 Features)")
dataset_choice = input("Digite o número correspondente ao dataset: ")

# Carregar o dataset de treinamento e teste
train, test = load_dataset(dataset_choice)

# Verificar se os datasets foram carregados corretamente
if train is not None and test is not None:
    # Separar as features (X) e o alvo (y) para treinamento e teste
    X_train = train.drop(columns=['class', 'ID']).values
    y_train = train['class'].values

    X_test = test.drop(columns=['class', 'ID']).values
    y_test = test['class'].values

    # Interação com o usuário para escolher o valor de k
    k = int(input("Escolha o valor de k (número de vizinhos): "))

    # Executar o KNN com o valor de k escolhido
    knn = run_knn(X_train, y_train, X_test, y_test, k)

    # Pergunta 2: Analisar os vizinhos mais próximos para N1
    instance_idx = 0  # N1 é a primeira instância de teste
    list_nearest_neighbors(knn, X_test, instance_idx, k)

    # Pergunta 4: Perturbações na instância N4 e análise do impacto
    n4_idx = 3  # N4 é a quarta instância de teste
    perturb_and_classify(knn, X_test, n4_idx, [0.3, 0.85], attribute=1)  # Atributo 1 é citric acid

    # Visualizar os vizinhos mais próximos e instâncias
    plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx, k)

else:
    print("Erro ao carregar os datasets.")
