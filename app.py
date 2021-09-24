
from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

model = load_model(os.getcwd()+'/models/deployment_finalize')

results = {0: 'trash', 1: 'good', 2: 'awesome'}


@app.route("/")
def home():
   return render_template("home.html")


@app.route("/predict", methods=['POST'])
def predict():
  features = [float(v) for _, v in request.form.items()]
  cols = [k for k, _ in request.form.items()]

  sample = np.array(features)
  sample_frame = pd.DataFrame([sample], columns=cols)

  prediction = predict_model(model, data=sample_frame)
  prediction = results[prediction.Label[0]]

  return render_template('result.html', pred=prediction)


if __name__ == '__main__':
    app.run()
