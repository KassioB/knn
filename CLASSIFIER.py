import os  # Módulo para manipular caminhos e arquivos no sistema operacional
import cv2  # Biblioteca OpenCV para processamento de imagens
import numpy as np  # Biblioteca para manipulação de arrays numéricos
import pandas as pd  # Biblioteca para manipulação de dados tabulares
from sklearn.model_selection import train_test_split  # Função para dividir dados em treino e teste
from sklearn.neighbors import KNeighborsClassifier  # Algoritmo K-Nearest Neighbors (KNN)
from sklearn.preprocessing import StandardScaler  # Normalizador de dados para melhor desempenho do modelo
from sklearn.decomposition import PCA  # Algoritmo de redução de dimensionalidade Principal Component Analysis (PCA)

# Carregar o CSV contendo os nomes das imagens e seus respectivos rótulos
df = pd.read_csv(r"cat_dog.csv", names=["image", "label"], skiprows=1)  
# O parâmetro 'names' define os nomes das colunas, e 'skiprows=1' pula a primeira linha caso seja um cabeçalho.

# Caminho da pasta onde estão as imagens
image_folder = r"cat_dog"

# Função para extrair características básicas de uma imagem
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Carrega a imagem em escala de cinza para reduzir complexidade
    image = cv2.resize(image, (64, 64))  # Redimensiona para um tamanho fixo de 64x64 pixels
    return image.flatten()  # Transforma a imagem em um vetor unidimensional para entrada no modelo

# Criar listas para armazenar características (features) e rótulos (labels)
features = []
labels = []

# Percorrer cada linha do DataFrame para processar as imagens
for _, row in df.iterrows():
    img_name = row["image"]  # Nome da imagem no CSV
    label = row["label"]  # 0 para gato, 1 para cachorro
    img_path = os.path.join(image_folder, img_name)  # Caminho completo da imagem

    if os.path.exists(img_path):  # Verifica se o arquivo de imagem realmente existe
        features.append(extract_features(img_path))  # Extrai características e adiciona à lista
        labels.append(label)  # Adiciona o rótulo correspondente

# Converter as listas para arrays numpy para facilitar o processamento
X = np.array(features)
y = np.array(labels)

# Normalizar os dados para que todas as características tenham a mesma escala
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aplicar PCA para reduzir a dimensionalidade dos dados e melhorar a eficiência do modelo
pca = PCA(n_components=100)  # Reduzindo para 100 componentes principais
X = pca.fit_transform(X)

# Dividir os dados em treino (80%) e teste (20%) de forma aleatória, mas reproduzível (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo KNN com k=7 (usando os 7 vizinhos mais próximos para classificação)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Testar o modelo e calcular a acurácia nos dados de teste
accuracy = knn.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")  # Exibe o resultado em percentual
