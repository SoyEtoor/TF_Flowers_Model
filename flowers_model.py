import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import urllib.request

# Descargar el dataset
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_path = tf.keras.utils.get_file("flower_photos.tgz", origin=url, extract=True)
base_dir = os.path.join(os.path.dirname(dataset_path), 'flower_photos')

# Definir parámetros de preprocesamiento
batch_size = 32
img_height = 180
img_width = 180

# Crear generadores para los datos de entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Cambia 5 por el número de clases en tu dataset
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 10  # Cambia el número de épocas según sea necesario
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Evaluar el modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Guardar en formato TensorFlow Serving
serving_model_path = os.path.join("flowers-model", "1")
tf.saved_model.save(model, serving_model_path)
print(f"Modelo guardado en formato TensorFlow Serving en {serving_model_path}")

# Guardar en formato Keras para pruebas y recarga
keras_model_path = "flowers-model-keras"
model.save(keras_model_path)
print(f"Modelo guardado en formato Keras en {keras_model_path}")

# Predicción de una nueva imagen
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch
    img_array = img_array / 255.0  # Normalizar
    return img_array

# Cargar y usar el modelo guardado en formato Keras para predicciones
loaded_model = tf.keras.models.load_model(keras_model_path)
img_path = 'rosa.jpg'  # Cambia esto a la ruta de tu imagen
img_array = load_and_preprocess_image(img_path)
predictions = loaded_model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f'Predicted class: {predicted_class}')

