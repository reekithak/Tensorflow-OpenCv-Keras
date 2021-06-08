from flask import Flask ,render_template , request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from model import predict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

app = Flask(__name__)

photos = UploadSet('photos',IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app,photos)

@app.route('/home',methods=['GET','POST'])
def home():
    welcome = "Hey People!"
    return welcome
@app.route('/',methods=['GET','POST'])
def base():
    new_txt = "uh Not quite here"
    return new_txt

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST' and  'photo' in request.files:
        filename = photos.save(request.files['photo'])

        image = load_img('./static/img/'+filename,target_size=(224,224))

        prediction = predict(image)
    

        return prediction

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(port=5000,debug=True)