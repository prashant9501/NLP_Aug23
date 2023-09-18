# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:13:00 2023

@author: Prashant
"""

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model_pipeline = joblib.load('model_pipeline.pkl')

@app.route('/')  # this will be triggered when we land up into the root of the web application 
def home():
    return "<H2> Welcome to Sentiment PRediction Project! </H2>"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Error handling if nothing is sent
    if not request.json or 'tweet' not in request.json:
        return jsonify({'error': 'No data provided'}), 400

    tweet = request.json['tweet']
    print(tweet)
 
    prediction = model_pipeline.predict([tweet])
    
    # Convert binary prediction to sentiment
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    # return jsonify({'sentiment': sentiment})
    return "<H2> Tweet Sentiment: " + sentiment + "</H2>"

if __name__ == '__main__':
    app.run(debug=False)



