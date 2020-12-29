# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 19:13:44 2019

@author: ASPDISCOVERY
"""

import numpy as np
import pandas as pd

import json
import joblib

model=joblib.load('iris.pkl')
from flask import Flask,request,jsonify

app=Flask(__name__)

@app.route('/',methods=["POST"])
def predict():
    data=request.get_json(force=True)
    print(data)
    predi=model.predict(np.array([[data['a'],data['b'],data['c'],data['d']]]))
    output=predi
    print(output[0])
    
    return str(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)




