import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

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
        return None, None
    return train, test

def run_knn(X_train, y_train, X_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo com k={k}: {accuracy * 100:.2f}%')
    return knn

def list_nearest_neighbors(knn, X_test, instance_idx, k):
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    print(f"Vizinhos mais próximos da instância {instance_idx}:")
    print(f"Distâncias: {distances}")
    print(f"Índices: {indices}")

def perturb_and_classify(knn, X_test, original_instance, new_values, attribute):
    for val in new_values:
        perturbed_instance = X_test[original_instance].copy()
        perturbed_instance[attribute] = val
        pred_class = knn.predict(perturbed_instance.reshape(1, -1))
        print(f"Valor {val} para {attribute}: Classe prevista: {pred_class[0]}")

def plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx, k):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Treinamento', cmap='coolwarm', edgecolor='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', s=100, label='Teste', cmap='coolwarm', edgecolor='k')
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    plt.scatter(X_test[instance_idx, 0], X_test[instance_idx, 1], c='green', marker='x', s=200, label=f'Instância N{instance_idx+1}')
    for i in indices[0]:
        plt.plot([X_test[instance_idx, 0], X_train[i, 0]], [X_test[instance_idx, 1], X_train[i, 1]], 'k--', lw=1)
    plt.title(f'Vizinhos mais próximos para a instância N{instance_idx+1} com k={k}')
    plt.xlabel('Total Sulfur Dioxide')
    plt.ylabel('Citric Acid')
    plt.legend()
    if not os.path.exists('results'):
        os.makedirs('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/knn_results_instance_{instance_idx+1}_{timestamp}.png'
    plt.savefig(filename)
    print(f"Imagem salva como {filename}")

print("Escolha o dataset para Treinamento e Teste:")
print("1. Dados Originais (2 Features)")
print("2. Dados Normalizados (2 Features)")
print("3. Dados Originais (11 Features)")
print("4. Dados Normalizados (11 Features)")
dataset_choice = input("Digite o número correspondente ao dataset: ")

train, test = load_dataset(dataset_choice)

if train is not None and test is not None:
    X_train = train.drop(columns=['class', 'ID']).values
    y_train = train['class'].values
    X_test = test.drop(columns=['class', 'ID']).values
    y_test = test['class'].values
    k = int(input("Escolha o valor de k (número de vizinhos): "))
    knn = run_knn(X_train, y_train, X_test, y_test, k)

    for i in range(len(X_test)):
        list_nearest_neighbors(knn, X_test, i, k)
        plot_knn_results(X_train, y_train, X_test, y_test, knn, i, k)

    n4_idx = 3
    perturb_and_classify(knn, X_test, n4_idx, [0.3, 0.85], attribute=1)
else:
    print("Erro ao carregar os datasets.")
