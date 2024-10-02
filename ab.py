import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.copy())
    X_test_scaled = scaler.transform(X_test.copy())
    return X_train_scaled, X_test_scaled

def load_dataset(choice):
    try:
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
    except FileNotFoundError as e:
        print(f"Erro ao carregar os arquivos: {e}")
        return None, None
    return train.copy(), test.copy()

def run_knn(X_train, y_train, X_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return knn, accuracy

def list_nearest_neighbors_with_ids(knn, X_test, instance_idx, k, train_df):
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    neighbors_ids = train_df.iloc[indices[0]]['ID'].values
    neighbors_ids_sorted = sorted(neighbors_ids)
    neighbors_str = ";".join(neighbors_ids_sorted)
    print(f"Vizinhos mais próximos da instância N{instance_idx + 1}: {neighbors_str}")
    return neighbors_str

def plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx, k, feature_names, experiment_name, dataset_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Treinamento', cmap='coolwarm', edgecolor='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', s=100, label='Teste', cmap='coolwarm', edgecolor='k')
    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    plt.scatter(X_test[instance_idx, 0], X_test[instance_idx, 1], c='green', marker='x', s=200, label=f'Instância N{instance_idx+1}')
    for i in indices[0]:
        plt.plot([X_test[instance_idx, 0], X_train[i, 0]], [X_test[instance_idx, 1], X_train[i, 1]], 'k--', lw=1)
    plt.title(f'{experiment_name} com k={k} - {dataset_name}')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()), max(X_train[:, 0].max(), X_test[:, 0].max())
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()), max(X_train[:, 1].max(), X_test[:, 1].max())
    plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/{experiment_name}_{dataset_name}_k{k}_results_{timestamp}.png'
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(filename)
    print(f"Imagem salva como {filename}")

def experiment_A():
    print("\n--- Iniciando Experimento A ---")
    ks = [1, 3, 5, 7]
    datasets = [('1', 'Dados Originais (2 Features)'), ('2', 'Dados Normalizados (2 Features)')]
    for dataset_choice, dataset_name in datasets:
        train, test = load_dataset(dataset_choice)
        feature_names = train.drop(columns=['ID', 'class']).columns[:2]
        X_train = train.drop(columns=['ID', 'class']).values
        y_train = train['class'].values
        X_test = test.drop(columns=['ID', 'class']).values
        y_test = test['class'].values
        if dataset_choice == '2':
            X_train, X_test = standardize_data(X_train, X_test)
        print(f"\nDataset: {dataset_name}")
        for k in ks:
            knn, accuracy = run_knn(X_train, y_train, X_test, y_test, k)
            print(f"\n--- k={k} ---")
            print(f"Acurácia total = {accuracy:.2f}")
            plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx=0, k=k, feature_names=feature_names, experiment_name="Experiment_A", dataset_name=dataset_name)
            acertos = 0
            for instance_idx in range(4):
                predicted_class = knn.predict(X_test[instance_idx].reshape(1, -1))[0]
                true_class = y_test[instance_idx]
                if predicted_class == true_class:
                    acertos += 1
                neighbors_str = list_nearest_neighbors_with_ids(knn, X_test, instance_idx, k, train)
                print(f"Instância N{instance_idx + 1}: Prevista = {predicted_class}, Real = {true_class}, Vizinhos: {neighbors_str}")
            acuracia_individual = acertos / 4
            print(f"Acurácia para N1 a N4 com k={k}: {acuracia_individual:.2f}\n")

def experiment_B():
    print("\n--- Iniciando Experimento B ---")
    train, test = load_dataset('1')
    feature_names = train.drop(columns=['ID', 'class']).columns[:2]
    X_train = train.drop(columns=['ID', 'class']).values
    y_train = train['class'].values
    X_test = test.drop(columns=['ID', 'class']).values
    y_test = test['class'].values
    knn, _ = run_knn(X_train, y_train, X_test, y_test, k=5)
    plot_knn_results(X_train, y_train, X_test, y_test, knn, instance_idx=0, k=5, feature_names=feature_names, experiment_name="Experiment_B", dataset_name="Dados Originais (2 Features)")
    for instance_idx in range(4):
        neighbors_str = list_nearest_neighbors_with_ids(knn, X_test, instance_idx, k=5, train_df=train)
        print(f"IDs dos vizinhos mais próximos de N{instance_idx + 1}: {neighbors_str}")

def perturb_and_classify(knn, X_test, original_instance, new_values, attribute, model_name):
    for val in new_values:
        perturbed_instance = X_test[original_instance].copy()
        perturbed_instance[attribute] = val
        pred_class = knn.predict(perturbed_instance.reshape(1, -1))
        print(f"[{model_name}] Valor {val} para 'citric acid': Classe prevista: {pred_class[0]}")
        distances, indices = knn.kneighbors(perturbed_instance.reshape(1, -1), n_neighbors=5)
        print(f"[{model_name}] Vizinhos mais próximos (IDs): {indices[0]}")
        print(f"[{model_name}] Distâncias para vizinhos: {distances[0]}")

experiment_A()
experiment_B()
