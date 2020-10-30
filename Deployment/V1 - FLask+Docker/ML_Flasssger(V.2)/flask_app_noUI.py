from flask import Flask,request

import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
pickle_in = open('classifier.pkl','rb')
clf = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def pred_():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction = clf.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is " + str(prediction)

@app.route("/predict_file",methods=['POST'])
def pred_1():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = clf.predict(df_test)
    return "The predicted values are" + str(list(prediction))





if(__name__=='__main__'):
    app.run()