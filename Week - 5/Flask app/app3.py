# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:13:00 2023

@author: Prashant
"""

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model_pipeline = joblib.load('model_pipeline.pkl')

@app.route('/')  # this will be triggered when we land up into the root of the web application 
def home():
    return "<H2> Welcome to Sentiment PRediction Project! </H2>"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    # Extract the tweet from the URL's query parameters
    tweet = request.args.get('tweet')

    prediction = None

    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet:
            pred = model_pipeline.predict([tweet])
            prediction = 'Positive' if pred[0] == 1 else 'Negative'

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False)



