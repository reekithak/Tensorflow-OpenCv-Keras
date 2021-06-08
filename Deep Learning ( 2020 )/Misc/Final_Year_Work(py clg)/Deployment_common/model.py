import tensorflow.keras
import tensorflow as tf
import numpy as np
from flask import Flask
from tensorflow import keras
from tensorflow.keras.preprocessing import image



model = keras.models.load_model("Models\model_weights.h5")
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


def preprocess_image(local_img,target_size=(150, 150)):
    local_img = image.img_to_array(local_img)
    local_img = tf.image.rgb_to_grayscale(local_img)
    local_img = np.expand_dims(local_img, axis=0)
    local_img = local_img/255.0

    return local_img

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
