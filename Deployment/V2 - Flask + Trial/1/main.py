
import numpy as np
import flask 
import request , jsonify , render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    init_features = [int(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)
    output = round(final_features)

    return render_template("index.html",prediction_text="Exployee salary should be {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)