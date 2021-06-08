import tensorflow.keras
import tensorflow as tf
import numpy as np
from flask import Flask
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model("Models\keras_model.h5")

classes = ['diseased cotton leaf',
         'diseased cotton plant',
         'fresh cotton leaf',
         'fresh cotton plant',
         'Pepper__bell___Bacterial_spot',
         'Pepper__bell___healthy',
         'Potato___Early_blight',
         'Potato___healthy',
         'Potato___Late_blight',
         'Tomato___Bacterial_spot',
         'Tomato___Early_blight',
         'Tomato___healthy',
         'Tomato___Late_blight',
         'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot',
         'Tomato___Tomato_mosaic_virus',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
         ]
labels = ['Class 1',
         'Class 2',
         'Class 3',
         'Class 4',
         'Class 5',
         'Class 6',
         'Class 7',
         'Class 8',
         'Class 9',
         'Class 10',
         'Class 11',
         'Class 12',
         'Class 13',
         'Class 14',
         'Class 15',
         'Class 16',
         'Class 17',
         'Class 18',
         'Class 19'
         ]

def preprocess_image(image,target_size=(224, 224)):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, (224,224), Image.ANTIALIAS)
    image_array = np.asarray(image)
    image = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = image 
    
    return data

print("keras Model Loading")

def predict(image):
    processed_image = preprocess_image(image,target_size=(224,224))
    prediction = model.predict(processed_image)
    lcl = prediction.flatten()
    new_var = lcl.max()
    for index,item in enumerate(lcl):
        if item == new_var:
            class_name = classes[index]
    return {
        "Predicted Class ": class_name 
        }
