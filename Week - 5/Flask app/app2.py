# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:13:00 2023

@author: Prashant
"""

from flask import Flask, request, render_template_string
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

    # Error handling if no tweet is provided
    if not tweet:
        return "<H2> Error 400: No tweet provided </H2>"

    prediction = model_pipeline.predict([tweet])
    
    # Convert binary prediction to sentiment
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    # Return the result within an h2 HTML tag
    return render_template_string(f"<h2>Tweet Sentiment: {sentiment}</h2>")

    # return "<H2> Tweet Sentiment: " + sentiment + "</H2>"

if __name__ == '__main__':
    app.run(debug=False)



