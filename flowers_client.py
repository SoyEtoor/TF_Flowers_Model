import requests
import json
import cv2
import numpy as np

image_path = "rose.jpg"

size = (100, 100)  
img = cv2.imread(image_path)
if img is None:
    raise ValueError("No se pudo leer la imagen.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
img = cv2.resize(img, size) 
img = img.astype('float32') / 255.0  
img = np.expand_dims(img, axis=-1)  
img = np.expand_dims(img, axis=0)  

print(f"Forma de la imagen: {img.shape}")

payload = {
    "instances": img.tolist()  
}

url = "https://tensorflow-flowers-model-00oo.onrender.com/v1/models/flowers-model:predict"
response = requests.post(url, json=payload)

if response.status_code == 200:
    predictions = response.json()['predictions'][0]
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    print("Predic Clases: [daisy, tulip, dandilion, sunflower, rose]")
    print(f"Clase predicha: {predicted_class}, Confianza: {confidence:.2f}")
else:
    print(f"Error: {response.status_code}, {response.text}")
