from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from os.path import join
import tensorflow as tf
from os import listdir
import numpy as np
import pathlib
import random
import pandas
import cv2
import os

dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
directory = tf.keras.utils.get_file('flower_photos', origin=dataset, untar=True)
data = pathlib.Path(directory)
folders = os.listdir(data)

image_names = []
train_labels = []
train_images = []

size = 64,64
print('folders')
folders.remove("LICENSE.txt")
print(folders)

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue

train = np.array(train_images)
print(train.shape)

train = train.astype('float32') / 255.0
label_dummies = pandas.get_dummies(train_labels)
labels =  label_dummies.values.argmax(1)
pandas.unique(train_labels)
print(pandas.unique(labels))

union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

train = np.array(train)
labels = np.array(labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train,labels, epochs=5)

export_path = 'flowers-model/1/'
tf.saved_model.save(model, os.path.join('./',export_path))