import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# =====================================
# CONFIGURARE
# =====================================

# imagini pentru inferență + clustering
BASE_PATH = "/Users/alexandrabontidean/Desktop/Dataset/colored/images"

# date etichetate pentru antrenare
TRAIN_PATH = "/Users/alexandrabontidean/PycharmProjects/stoma/dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
N_CLUSTERS = 2   # HEALTHY vs CARIES


# =====================================
# 1. ÎNCĂRCARE DATE ANTRENARE
# =====================================

print("\n[INFO] Se încarcă datele de antrenare...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("[INFO] Clase detectate:", train_gen.class_indices)


# =====================================
# 2. DEFINIRE MODEL
# =====================================

print("\n[INFO] Se construiește modelul...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================================
# 3. ANTRENARE
# =====================================

print("\n[INFO] Pornesc antrenarea...")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("caries_detector_resnet50.h5")
print("[INFO] Model salvat: caries_detector_resnet50.h5")


# =====================================
# 4. ÎNCĂRCARE IMAGINI PENTRU TESTARE
# =====================================

def load_images(folder, size):
    images, names = [], []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, f)
            img = Image.open(path).convert("RGB").resize(size)
            images.append(np.array(img))
            names.append(f)
    return np.array(images), names


print("\n[INFO] Se încarcă imaginile pentru analiză...")
images, image_names = load_images(BASE_PATH, IMG_SIZE)
print(f"[INFO] {len(images)} imagini încărcate")


# =====================================
# 5. EXTRAGERE FEATURE-URI
# =====================================

X = preprocess_input(images)

feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-3].output  # Dense(256)
)

features = feature_extractor.predict(X)
features_norm = normalize(features)


# =====================================
# 6. CLUSTERING KMEANS
# =====================================

print("\n[INFO] Pornesc clustering-ul...")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features_norm)

for c in range(N_CLUSTERS):
    print(f"Cluster {c}: {np.sum(cluster_labels == c)} imagini")


# =====================================
# 7. AFIȘARE CLUSTERE
# =====================================

for c in range(N_CLUSTERS):
    idx = np.where(cluster_labels == c)[0]

    if len(idx) == 0:
        print(f"[WARN] Cluster {c + 1} este gol.")
        continue

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Cluster {c + 1} - {len(idx)} imagini", fontsize=16)

    for i, j in enumerate(idx[:8]):
        plt.subplot(1, 8, i + 1)
        plt.imshow(images[j])
        plt.axis("off")
        plt.title(image_names[j], fontsize=7)

    plt.show()


# =====================================
# 8. CLASIFICARE CARIES vs HEALTHY (CORECT)
# =====================================

print("\n[INFO] Clasificare finală cu CNN...")

preds = model.predict(X)
pred_labels = (preds > 0.5).astype(int).ravel()

for label, name in zip(pred_labels[:10], image_names[:10]):
    cls = "CARIES" if label == 1 else "HEALTHY"
    print(f"{name} → {cls}")

print("\nEXECUȚIE FINALIZATĂ CU SUCCES")
