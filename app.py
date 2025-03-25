#!/usr/bin/env python
# coding: utf-8

# In[9]:


#app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
#Initialize Flask app
app= Flask(_name_)
#Load trained model
model = joblib.load('iris_model_pk1')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    #Get data from form
    try:
        features=[float(request.form[f'feature{i}'])for i in range(1,5)]
    except ValueError:
        return render_template('result.html',prediction="Inavlid input. Please enter numeric values.")
        #Make prediction
        prediction=model.predict([features])[0]
        #Map prediction to class name
        class_names=['Setosa','Versicolor','Virginica']
        result=class_names[prediction]
        return render_template('result.html',prediction=result)
if__name__=='__main__':
    app.run(debug=True)


# In[ ]:




