from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import pathlib

dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True)
data = pathlib.Path(directory)
folders = sorted(os.listdir(data))

image_names = []
train_labels = []
train_images = []

size = (64, 64)

print(f"Clases encontradas: {folders}")

for folder in folders:
    folder_path = os.path.join(data, folder)
    print(f"Procesando carpeta: {folder}")
    for file in os.listdir(folder_path):
        if file.endswith("jpg"):
            image_path = os.path.join(folder_path, file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Advertencia: No se pudo leer la imagen {image_path}.")
                continue
            img = cv2.resize(img, size)
            train_images.append(img)
            train_labels.append(folders.index(folder))  # Usar índices como etiquetas

if not train_images:
    raise ValueError("No se encontraron imágenes procesadas. Verifica las rutas y el contenido del dataset.")

train = np.array(train_images, dtype='float32') / 255.0
labels = np.array(train_labels)

print(f"Imágenes procesadas: {train.shape}, Etiquetas únicas: {set(labels)}")

train_images, val_images, train_labels, val_labels = train_test_split(
    train, labels, test_size=0.2, random_state=42
)

# Construir el modelo (CNN)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(folders), activation='softmax')  
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stopping])

export_path = 'flowers-model/1/'
os.makedirs(os.path.dirname(export_path), exist_ok=True)
tf.saved_model.save(model, os.path.join('./', export_path))

print("Modelo guardado correctamente.")

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen en {image_path}")
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)
    return predicted_class, confidence

sample_image = os.path.join(data, folders[0], os.listdir(os.path.join(data, folders[0]))[0])
predicted_class, confidence = predict_image(model, sample_image)
print(f"Predicción: Clase {folders[predicted_class]}, Confianza: {confidence:.2f}")
