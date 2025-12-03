# ======================
# 0. Importuri generale
# ======================
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from keras.applications.resnet50 import ResNet50, preprocess_input

# ======================
# 1. Setări cale și parametri
# ======================
BASE_PATH = "/Users/alexandrabontidean/Desktop/Dataset/colored/images"
IMAGE_SIZE = (224, 224)  # dimensiune cerută de ResNet50
N_CLUSTERS = 3  # poți modifica numărul de cluster-e

# ======================
# 2. Încarcă imaginile
# ======================
def load_images(folder_path, image_size=(224,224)):
    img_list = []
    img_names = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            img = img.resize(image_size)
            img_list.append(np.array(img))
            img_names.append(fname)
        except:
            print(f"Nu am putut încărca {fname}")
    return np.array(img_list), img_names

images, image_names = load_images(BASE_PATH, IMAGE_SIZE)
print(f"Am încărcat {len(images)} imagini.")

# ======================
# 3. Extrage feature-uri cu ResNet50
# ======================
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # vector 2048 dim
X = preprocess_input(images)  # normalizare
features = base_model.predict(X, verbose=1)
print("Am extras feature-urile.")

# ======================
# 4. Clustering K-Means
# ======================
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features)
print("Clustering efectuat.")

# ======================
# 5. Vizualizare imagini pe cluster
# ======================
for cluster in range(N_CLUSTERS):
    cluster_idx = np.where(labels == cluster)[0]
    plt.figure(figsize=(12,4))
    plt.suptitle(f"Cluster {cluster+1} - {len(cluster_idx)} imagini", fontsize=16)
    for i, idx in enumerate(cluster_idx):
        if i >= 10:  # afișează maxim 10 imagini per cluster
            break
        plt.subplot(1, 10, i+1)
        plt.imshow(images[idx].astype(np.uint8))
        plt.axis('off')
        plt.title(image_names[idx], fontsize=8)
    plt.show()
