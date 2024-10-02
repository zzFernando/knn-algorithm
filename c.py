import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_dataset(choice):
    try:
        if choice == '2':
            train = pd.read_csv('Dados_Normalizados_2Features/TrainingData_2F_Norm.txt', sep='\t')
            test = pd.read_csv('Dados_Normalizados_2Features/TestingData_2F_Norm.txt', sep='\t')
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

def display_neighbors_info(knn, X_test, original_instance, k, model_name):
    distances, indices = knn.kneighbors(X_test[original_instance].reshape(1, -1), n_neighbors=k)
    print(f"\n[{model_name}] Detalhes dos {k} vizinhos mais próximos da instância N{original_instance + 1}:")
    for i in range(k):
        print(f"Vizinhos {i + 1}: Índice {indices[0][i]}, Distância: {distances[0][i]}")

def plot_neighbors_11features_pca(X_train, y_train, X_test, knn, instance_idx, k, citric_acid_value, perturbation=False):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, marker='o', label='Treinamento', cmap='coolwarm', edgecolor='k')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], marker='s', s=100, label='Teste', edgecolor='k')

    label = f'N4 Perturbada (citric acid = {citric_acid_value})' if perturbation else f'Instância N4'
    plt.scatter(X_test_pca[instance_idx, 0], X_test_pca[instance_idx, 1], c='green', marker='x', s=200, label=label)

    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    for i in indices[0]:
        plt.plot([X_test_pca[instance_idx, 0], X_train_pca[i, 0]], [X_test_pca[instance_idx, 1], X_train_pca[i, 1]], 'k--', lw=1)

    plt.title(f'Vizinhos Mais Próximos (k={k}) - citric acid = {citric_acid_value}')
    plt.legend()

    if not os.path.exists('results'):
        os.makedirs('results')

    perturbation_str = 'perturbado' if perturbation else 'original'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/N4_{perturbation_str}_citricacid_{citric_acid_value}_pca_k{k}_{timestamp}.png'
    plt.savefig(filename)
    print(f"Imagem salva como {filename}")

def plot_neighbors_2features(X_train, y_train, X_test, knn, instance_idx, k, citric_acid_value, perturbation=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Treinamento', cmap='coolwarm', edgecolor='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='s', s=100, label='Teste', edgecolor='k')

    label = f'N4 Perturbada (citric acid = {citric_acid_value})' if perturbation else f'Instância N4'
    plt.scatter(X_test[instance_idx, 0], X_test[instance_idx, 1], c='green', marker='x', s=200, label=label)

    distances, indices = knn.kneighbors(X_test[instance_idx].reshape(1, -1), n_neighbors=k)
    for i in indices[0]:
        plt.plot([X_test[instance_idx, 0], X_train[i, 0]], [X_test[instance_idx, 1], X_train[i, 1]], 'k--', lw=1)

    plt.title(f'Vizinhos Mais Próximos (k={k}) - citric acid = {citric_acid_value}')
    plt.legend()

    if not os.path.exists('results'):
        os.makedirs('results')

    perturbation_str = 'perturbado' if perturbation else 'original'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/N4_{perturbation_str}_citricacid_{citric_acid_value}_2features_k{k}_{timestamp}.png'
    plt.savefig(filename)
    print(f"Imagem salva como {filename}")

def experiment_C():
    print("\n--- Iniciando Experimento C ---")
    train_2f, test_2f = load_dataset('2')
    train_11f, test_11f = load_dataset('4')

    if train_2f is None or test_2f is None or train_11f is None or test_11f is None:
        print("Erro ao carregar os datasets para o experimento C")
        return

    X_train_2f = train_2f.drop(columns=['ID', 'class']).values
    y_train_2f = train_2f['class'].values
    X_test_2f = test_2f.drop(columns=['ID', 'class']).values
    y_test_2f = test_2f['class'].values

    X_train_11f = train_11f.drop(columns=['ID', 'class']).values
    y_train_11f = train_11f['class'].values
    X_test_11f = test_11f.drop(columns=['ID', 'class']).values
    y_test_11f = test_11f['class'].values

    knn_2f, accuracy_2f = run_knn(X_train_2f, y_train_2f, X_test_2f, y_test_2f, k=5)
    knn_11f, accuracy_11f = run_knn(X_train_11f, y_train_11f, X_test_11f, y_test_11f, k=5)

    print(f"Acurácia M2 (2 features): {accuracy_2f:.4f}")
    print(f"Acurácia M11 (11 features): {accuracy_11f:.4f}")

    n4_idx = 3
    perturb_values = [0.3, 0.85]

    print("\n--- M2 (2 Features): N4 Original ---")
    plot_neighbors_2features(X_train_2f, y_train_2f, X_test_2f, knn_2f, n4_idx, k=5, citric_acid_value=1.0, perturbation=False)
    display_neighbors_info(knn_2f, X_test_2f, n4_idx, k=5, model_name="M2 (2 Features)")

    for val in perturb_values:
        print(f"\n--- M2 (2 Features): N4 Perturbada (citric acid = {val}) ---")
        X_test_2f_perturbed = X_test_2f.copy()
        X_test_2f_perturbed[n4_idx, 1] = val
        plot_neighbors_2features(X_train_2f, y_train_2f, X_test_2f_perturbed, knn_2f, n4_idx, k=5, citric_acid_value=val, perturbation=True)
        display_neighbors_info(knn_2f, X_test_2f_perturbed, n4_idx, k=5, model_name="M2 (2 Features)")

    print("\n--- M11 (11 Features): N4 Original ---")
    plot_neighbors_11features_pca(X_train_11f, y_train_11f, X_test_11f, knn_11f, n4_idx, k=5, citric_acid_value=1.0, perturbation=False)
    display_neighbors_info(knn_11f, X_test_11f, n4_idx, k=5, model_name="M11 (11 Features)")

    for val in perturb_values:
        print(f"\n--- M11 (11 Features): N4 Perturbada (citric acid = {val}) ---")
        X_test_11f_perturbed = X_test_11f.copy()
        X_test_11f_perturbed[n4_idx, 1] = val
        plot_neighbors_11features_pca(X_train_11f, y_train_11f, X_test_11f_perturbed, knn_11f, n4_idx, k=5, citric_acid_value=val, perturbation=True)
        display_neighbors_info(knn_11f, X_test_11f_perturbed, n4_idx, k=5, model_name="M11 (11 Features)")

experiment_C()
