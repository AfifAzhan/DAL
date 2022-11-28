# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:31:46 2022

@author: RAN
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
app = Flask(__name__)
# Load the model
model = pickle.load(open('model3.pkl','rb'))

df = pd.read_excel('svm_imp_feat_unbal.xlsx')
x = df.iloc[:, :9]
y = df.iloc[:, 9]

sc = StandardScaler()

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

x_train_sd = sc.fit_transform(x_train)
x_test_sd = sc.fit_transform(x_test)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    
    
    print(int_features[0])
    standardized_input =[]
     
    i = 0
    cols = x.columns
    for col in cols:
        inputStd = (int_features[i] - x_train[col].mean())/(x_train[col].std())
        i+=1
        standardized_input.append(inputStd)

    final_features = [np.array(standardized_input)]
    prediction = model.predict(final_features)
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Type of date: {}'.format(output))
    
    
@app.route('/api/',methods=['POST'])
def predict_api():
    #Direct API calls through request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data['exp'].values()))])
    
    output = prediction[0]
    return jsonify(output)
    

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    
    
    
    