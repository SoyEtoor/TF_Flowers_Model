from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import pathlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Descargar y descomprimir el dataset
dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True)
data = pathlib.Path(directory)
folders = [folder for folder in sorted(os.listdir(data)) if os.path.isdir(os.path.join(data, folder))]

print(f"Clases encontradas: {folders}")

# Preprocesar imágenes y etiquetas
train_images = []
train_labels = []
size = (128, 128)  # Aumentamos la resolución para MobileNetV2

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
            train_labels.append(folders.index(folder))  

if not train_images:
    raise ValueError("No se encontraron imágenes procesadas. Verifica las rutas y el contenido del dataset.")

train = np.array(train_images, dtype='float32') / 255.0
labels = np.array(train_labels)

print(f"Imágenes procesadas: {train.shape}, Etiquetas únicas: {set(labels)}")

# Dividir datos en entrenamiento y validación
train_images, val_images, train_labels, val_labels = train_test_split(
    train, labels, test_size=0.2, random_state=42
)

# Aumentación de datos
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = data_gen.flow(train_images, train_labels, batch_size=32)

# Construir modelo (Transfer Learning con MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congela las capas base

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),  # Regularización para evitar overfitting
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(folders), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=(val_images, val_labels),
    epochs=30,
    callbacks=[early_stopping]
)

# Guardar modelo
export_path = 'flowers-model/1/'
os.makedirs(os.path.dirname(export_path), exist_ok=True)
tf.saved_model.save(model, os.path.join('./', export_path))

print("Modelo guardado correctamente.")

# Función para predecir imágenes
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

# Probar predicción
sample_image = os.path.join(data, folders[0], os.listdir(os.path.join(data, folders[0]))[0])
predicted_class, confidence = predict_image(model, sample_image)
print(f"Predicción: Clase {folders[predicted_class]}, Confianza: {confidence:.2f}")

# Visualizar historia de entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.legend()
plt.show()
