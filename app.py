from flask import Flask, request, render_template,jsonify
import json
import numpy as np
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd


app = Flask(__name__)

def load_models():
    model = pickle.load(open("model/churn_pred.pkl", "rb"))
    return model


def predict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    # load model
    print(to_predict)
    model = load_models()
    prediction = model.predict(to_predict)
    response = json.dumps({'response': str(prediction)})
    print(int(prediction))
    return int(prediction)

@app.route('/')
@cross_origin()
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        print(to_predict_list)
        result = predict(to_predict_list)
        if result == 0:
            response = 'Congratulations! The customer will stay with the company.'
        else:
            response = 'There is a probability that the customer can churn.'
        return render_template("index.html", prediction=response)
    else:
        return render_template("index.html")



