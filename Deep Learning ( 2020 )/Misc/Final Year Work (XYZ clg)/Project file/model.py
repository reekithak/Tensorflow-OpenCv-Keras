import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from flask import request
from flask import jsonify
from flask import Flask
from keras_preprocessing import image
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf



model = load_model("Models\model_only.h5")
model.load_weights("Models\model_only.hdf5")
classes = ['glioma_tumor',
         'meningioma_tumor',
         'no_tumor',
         'pituitary_tumor']


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
