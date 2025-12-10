import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from keras.applications.resnet50 import ResNet50, preprocess_input

BASE_PATH = "/Users/alexandrabontidean/Desktop/Dataset/colored/images"
IMAGE_SIZE = (224, 224)
N_CLUSTERS = 3

def load_images(folder_path, image_size=(224, 224)):
    img_list = []
    img_names = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        fpath = os.path.join(folder_path, fname)

        try:
            img = Image.open(fpath).convert("RGB")
            img = img.resize(image_size)
            img_np = np.array(img).astype("float32")
            img_list.append(img_np)
            img_names.append(fname)

        except Exception as e:
            print(f"nu s-a putut incarcaa {fname}: {e}")

    return np.array(img_list), img_names


print("se incarca imaginile...")
images, image_names = load_images(BASE_PATH, IMAGE_SIZE)
print(f"s-au incarcat {len(images)} imagini.")


print("se proceseaza imaginile...")
X = preprocess_input(images)
print("shape X:", X.shape, X.dtype)

print("incarc ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

print("pornesc extragerea feature-urilor...")
features = base_model.predict(X, verbose=1)
print("s-au extras feature-urile:", features.shape)

#normalizarea
features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

print("pornesc clustering-ul...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
labels = kmeans.fit_predict(features_norm)
print("clustering efectuat.")

print("se afiseaza rezultatele clustering-ului...")

for cluster in range(N_CLUSTERS):
    cluster_idx = np.where(labels == cluster)[0]

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Cluster {cluster + 1} - {len(cluster_idx)} imagini", fontsize=16)

    for i, idx in enumerate(cluster_idx[:10]):
        plt.subplot(1, 10, i + 1)
        plt.imshow(images[idx].astype(np.uint8))
        plt.axis("off")
        plt.title(image_names[idx], fontsize=7)

    plt.show()

print("gata!!!!!")
