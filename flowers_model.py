import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Download the flower dataset
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_path = tf.keras.utils.get_file("flower_photos.tgz", origin=url, extract=True)
base_dir = os.path.join(os.path.dirname(dataset_path), 'flower_photos')

# List of flower classes
flower_classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Prepare the dataset
TAMANO_IMG = 100  # Image size

# Create dataset from directory
def load_and_preprocess_image(file_path, label):
    # Read the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)  # Convert to grayscale
    img = tf.image.resize(img, [TAMANO_IMG, TAMANO_IMG])
    img = img / 255.0  # Normalize
    return img, label

# Create image dataset
image_paths = []
labels = []

for i, flower_class in enumerate(flower_classes):
    class_dir = os.path.join(base_dir, flower_class)
    for img_path in os.listdir(class_dir):
        full_path = os.path.join(class_dir, img_path)
        image_paths.append(full_path)
        labels.append(i)

# Convert to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image)

# Prepare data for training
dataset_list = list(dataset)
X = np.array([img.numpy() for img, _ in dataset_list])
y = np.array([label.numpy() for _, label in dataset_list])

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=len(flower_classes))

# Split the data
split_index = int(len(X) * 0.85)
X_entrenamiento = X[:split_index]
X_validacion = X[split_index:]
y_entrenamiento = y[:split_index]
y_validacion = y[split_index:]

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(X_entrenamiento)

# Best performing model (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(len(flower_classes), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare data generator for training
data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

# Train model
model.fit(
    data_gen_entrenamiento,
    epochs=100, 
    batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32)))
)

# Evaluate the model
evaluation = model.evaluate(X_validacion, y_validacion)
print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

# Create export directory
export_dir = 'flowers-model/1'
os.makedirs(export_dir, exist_ok=True)

# Save the model in TensorFlow Serving format
tf.saved_model.save(model, export_dir)

# Verify the saved model
print(f"\nModel saved to: {export_dir}")
print("Contents of the export directory:")
for root, dirs, files in os.walk(export_dir):
    for file in files:
        print(os.path.join(root, file))

# Optional: Save class names for reference
with open(os.path.join(export_dir, 'class_names.txt'), 'w') as f:
    for cls in flower_classes:
        f.write(f"{cls}\n")
