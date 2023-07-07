# Importar as bibliotecas necessárias
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Importar mais classificadores diferentes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def load_data(path, fixed_size):
    """Carregar e processar as imagens do dataset.

    Args:
        path (str): O caminho para o diretório que contém as pastas com as imagens.
        fixed_size (tuple): O tamanho fixo para redimensionar as imagens.

    Returns:
        tuple: Dois arrays numpy, um com as imagens e outro com os rótulos.
    """
    # Criar listas vazias para armazenar as imagens e os rótulos
    images = []
    labels = []

    # Iterar pelas pastas e imagens do dataset
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            # Carregar a imagem em escala de cinza
            img = cv2.imread(os.path.join(path, folder, file), cv2.IMREAD_GRAYSCALE)

            # Redimensionar a imagem para o tamanho fixo
            img = cv2.resize(img, fixed_size)

            # Adicionar a imagem e o rótulo nas listas
            images.append(img)
            labels.append(folder)

    # Converter as listas para arrays numpy para facilitar a manipulação
    images = np.array(images)
    labels = np.array(labels)

    return images, labels




def encode_labels(labels):
    """Codificar os rótulos em classes únicas.

    Args:
        labels (array): O array com os rótulos originais.

    Returns:
        array: O array com os rótulos codificados.
    """
    # Codificar os rótulos (COVID-19: 0, Normal: 1, Pneumonia-Bacterial: 2, Pneumonia-Viral: 3)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    return labels



def extract_features(images):
    """Extrair características HOG, textura e de forma das imagens.

    Args:
        images (array): O array com as imagens.

    Returns:
        array: O array com as características HOG, textura e de forma.
    """
    # Definir uma função auxiliar para extrair características HOG de uma imagem
    def hog_features(image):
        features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=False)
        return features

    # Definir uma função auxiliar para extrair características de forma de uma imagem
    def shape_features(image):
        # Converter a imagem para binária
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Computar os momentos da imagem
        moments = cv2.moments(image)
        # Computar os momentos Hu da imagem
        hu_moments = cv2.HuMoments(moments)
        # Normalizar os momentos Hu para torná-los invariantes à escala e rotação
        hu_moments = np.log(np.abs(hu_moments))
        return hu_moments.flatten()
    
    def texture_features(image):
        # Calcular o padrão binário local da imagem com 8 vizinhos e raio 1
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        # Calcular o histograma do padrão binário
        hist = np.histogram(lbp, bins=10, range=(0, 10))[0]
        # Normalizar o histograma
        hist = hist / hist.sum()
        return hist

    # Extrair características HOG e de forma de todas as imagens usando as funções auxiliares
    hog_shape_features = np.array([np.concatenate([hog_features(img), shape_features(img), texture_features(img)]) for img in images])

    return hog_shape_features


def split_data(X, y, test_size, random_state):
    """Dividir o dataset em conjuntos de treino e teste.

    Args:
        X (array): O array com as características das imagens.
        y (array): O array com os rótulos das imagens.
        test_size (float): A proporção do dataset que será usada para o conjunto de teste.
        random_state (int): A semente aleatória para garantir a reprodutibilidade da divisão.

    Returns:
        tuple: Quatro arrays numpy, um com as características de treino, outro com as características de teste,
               outro com os rótulos de treino e outro com os rótulos de teste.
    """
    # Dividir o dataset em conjuntos de treino e teste usando a função train_test_split do sklearn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test



def evaluate_model(model, X_test, y_test, dim="multi"):
    """Avaliar o modelo com os dados de teste.

    Args:
        model (object): O modelo XGBoost treinado.
        X_test (array): O array com as características de teste.
        y_test (array): O array com os rótulos de teste.

    Returns:
        None
    """
    # Fazer previsões no conjunto de teste usando o modelo treinado
    predictions = model.predict(X_test)
    
    if(dim == "1d"):

        # Imprimir um relatório de classificação com as principais métricas (precisão, recall, f1-score)
        print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

        # Calcular a matriz de confusão com os rótulos verdadeiros e previstos
        confusion = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    else:
        
        # Imprimir um relatório de classificação com as principais métricas (precisão, recall, f1-score)
        print(classification_report(y_test, predictions))

        # Calcular a matriz de confusão com os rótulos verdadeiros e previstos
        confusion = confusion_matrix(y_test, predictions)
        
 

    if(dim == "1d"):
        # Calcular a acurácia do modelo comparando os rótulos verdadeiros e previstos
        accuracy = accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1))
    else:
            # Calcular a acurácia do modelo comparando os rótulos verdadeiros e previstos
        accuracy = accuracy_score(y_test, predictions)
    
    # Imprimir a acurácia usando uma f-string
    print(f"Acurácia: {accuracy:.2f}")
    
       
    # Plotar a matriz de confusão usando a biblioteca seaborn
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

def main():
    """Executar o programa principal."""
    
    # Definir o caminho para o dataset e o tamanho fixo das imagens
    path = 'img'
    fixed_size = (64, 64)

    # Carregar e processar as imagens do dataset
    images, labels = load_data(path, fixed_size)

    # Codificar os rótulos em vetores binários
    labels = encode_labels(labels)

    # Extrair características HOG das imagens
    hog_features = extract_features(images)

    # Dividir o dataset em conjuntos de treino e teste usando uma proporção de 80/20 e uma semente aleatória de 42
    X_train, X_test, y_train, y_test = split_data(
        hog_features, labels, test_size=0.2, random_state=42)

    
    # Definir uma lista com os nomes, os modelos e os hiperparâmetros dos classificadores
    classifiers = [
        ('XGBoost', XGBClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
        ('SVM', SVC(), {'kernel': ['rbf', 'linear'], 'C': [0.1, 1.0, 10.0]}),
        ('MLP', MLPClassifier(), {'hidden_layer_sizes': [(100,), (200,), (300,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'max_iter': [200, 200, 200], 'alpha': [0.3, 0.2, 0.1], 'learning_rate_init': [0.1, 0.01, 0.001]})
    ]

    # Pedir ao usuário que digite o nome do classificador que ele quer rodar
    choice = input("Digite o nome do classificador que você quer rodar (XGBoost, KNN, SVM ou MLP): ")

    # Usar um if-elif-else para executar o código correspondente ao classificador escolhido
    if choice == "XGBoost":
        # Fazer o tunning e avaliar o modelo XGBoost
        name, model, params = classifiers[0]
        print(f"Fazendo o tunning e avaliando o modelo {name}...")
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        evaluate_model(grid.best_estimator_, X_test, y_test)
    elif choice == "KNN":
        # Fazer o tunning e avaliar o modelo KNN
        name, model, params = classifiers[1]
        print(f"Fazendo o tunning e avaliando o modelo {name}...")
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        evaluate_model(grid.best_estimator_, X_test, y_test) 
    elif choice == "SVM":
        # Fazer o tunning e avaliar o modelo SVM
        name, model, params = classifiers[2]
        print(f"Fazendo o tunning e avaliando o modelo {name}...")
        grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        evaluate_model(grid.best_estimator_, X_test, y_test)
    elif choice == "MLP":
        # Fazer o tunning e avaliar o modelo MLP
        name, model, params = classifiers[3]
        print(f"Avaliando o modelo {name}...")
        #grid = GridSearchCV(model, params, scoring='accuracy', cv=5)
        model.fit(X_train, y_train)
        #print(f"Melhores hiperparâmetros: {grid.best_params_}")
        evaluate_model(model, X_test, y_test)
    else:
        # Imprimir uma mensagem de erro se o usuário digitar um nome inválido
        print("Nome inválido. Por favor digite um dos nomes válidos: XGBoost, KNN, SVM ou MLP.")


if __name__ == "__main__":
   main()
