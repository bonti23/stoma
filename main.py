import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


BASE_PATH = "/Users/alexandrabontidean/Desktop/Dataset/colored/images"

TRAIN_PATH = "/Users/alexandrabontidean/PycharmProjects/stoma/dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
N_CLUSTERS = 3

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

print("[INFO] Clase:", train_gen.class_indices)

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

print("\n[INFO] Pornesc antrenarea...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("caries_detector_resnet50.h5")
print("[INFO] Model salvat ca caries_detector_resnet50.h5")


def load_images(folder, size):
    imgs = []
    names = []

    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(size)
                imgs.append(np.array(img))
                names.append(fname)
            except Exception as e:
                print(f"Eroare la {fname}: {e}")

    return np.array(imgs), names


print("\n[INFO] Se încarcă imaginile pentru clustering...")
images, image_names = load_images(BASE_PATH, IMG_SIZE)
print(f"[INFO] {len(images)} imagini încărcate")


print("\n[INFO] Extrag feature-uri...")

X = preprocess_input(images)

feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-3].output
)

features = feature_extractor.predict(X)
features_norm = normalize(features)

print("\n[INFO] Pornesc clustering-ul...")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(features_norm)

print("[INFO] Clustering finalizat!")


for c in range(N_CLUSTERS):
    idx = np.where(cluster_labels == c)[0]

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Cluster {c + 1} - {len(idx)} imagini", fontsize=16)

    for i, j in enumerate(idx[:8]):
        plt.subplot(1, 8, i + 1)
        plt.imshow(images[j].astype(np.uint8))
        plt.axis("off")
        plt.title(image_names[j], fontsize=7)

    plt.show()