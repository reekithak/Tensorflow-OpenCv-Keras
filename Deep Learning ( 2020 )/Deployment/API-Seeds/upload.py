from flask import Flask ,render_template , request
from flask_uploads import UploadSet, configure_uploads, IMAGES

from model import predict
from tensorflow.keras.preprocessing.image import load_img

app = Flask(__name__)

photos = UploadSet('photos',IMAGES)


app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app,photos)

@app.route('/home',methods=['GET','POST'])

def home():
    welcome = "Hey People!"
    return welcome


@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST' and  'photo' in request.files:
        filename = photos.save(request.files['photo'])

        image = load_img('./static/img/'+filename,target_size=(224,224))

        prediction = predict(image)
        answer = "Excellent:{}\nGood:{}\nAverage:{}\nBad:{}\nWorst:{}".format(prediction[0][0],
        prediction[0][1],
        prediction[0][2],
        prediction[0][3],
        prediction[0][4])

        return answer

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)