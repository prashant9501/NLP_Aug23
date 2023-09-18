# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:13:00 2023

@author: Prashant
"""

from flask import Flask, request, jsonify
import joblib
# import contractions
# import nltk
# import re
# from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# # Load your trained model
# model = joblib.load('sentiment_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# final_model_pipeline = joblib.load('final_model_pipeline.pkl')

model_pipeline = joblib.load('model_pipeline.pkl')

# Define the Data Cleaner Function (assuming you already have it from your project)
def data_cleaner(raw_tweet):
    '''
    This function cleans the raw tweet
    '''
    #Substituting contractions
    cleaned_tweet = contractions.fix(raw_tweet)

    # User-mentions Removal
    cleaned_tweet = re.sub("@[A-Za-z0-9]+", "", cleaned_tweet)

    # Hashtag Removal
    cleaned_tweet = re.sub("#", "", cleaned_tweet)

    #Hyperlink Removal
    cleaned_tweet = re.sub(r"http\S+","", cleaned_tweet)

    # Punctuation, Special Characters and digits Removal (Retaining only the alphabets)
    cleaned_tweet = re.sub(r"[^a-zA-Z]", " " , cleaned_tweet )
    
    # convert the tweet into lowercase & get rid of any leading or trailing spaces
    cleaned_tweet = cleaned_tweet.lower().strip()  

    # Retain only those token which have length > 2 characters
    cleaned_tweet = [token for token in cleaned_tweet.split() if len(token)>2]
    
    new_sent = ''
    for token in cleaned_tweet:
        new_sent = new_sent + lemmatizer.lemmatize(token) + ' '
    
    return new_sent.strip()



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
    # cleaned_tweet = data_cleaner(tweet)

    # Convert cleaned tweet to TFIDF format
    # tfidf_tweet = vectorizer.transform([cleaned_tweet])
    # Predict using the model
    # prediction = model.predict(tfidf_tweet)
    # prediction = final_model_pipeline.predict([cleaned_tweet])
    
    prediction = model_pipeline.predict([tweet])
    
    # Convert binary prediction to sentiment
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    # return jsonify({'sentiment': sentiment})
    return "<H2> Tweet Sentiment: " + sentiment + "</H2>"

if __name__ == '__main__':
    app.run(debug=False)



