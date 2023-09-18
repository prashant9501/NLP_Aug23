#!/usr/bin/env python
# coding: utf-8

# ## Load the Tweets Dataset

# In[20]:


import nltk
import matplotlib.pyplot as plt
import contractions
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


import pandas as pd
pd.set_option("display.max_colwidth", 200)


# In[15]:


# tweets = pd.read_pickle("cleaned_tweets_v1.pkl")
tweets = pd.read_csv("tweets.csv")
tweets.head(5)


# In[4]:


tweets.shape


# In[16]:


# LEt's map 0 as 1, and 1 as 0
tweets['label'] = tweets['label'].map({0: 1, 1: 0})
tweets.head()


# In[17]:


# drop the id column
tweets.drop("id", axis=1, inplace=True)


# In[18]:


tweets.head()


# In[19]:


def tweet_cleaner(raw_tweet):
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
    lemmatizer = WordNetLemmatizer()
    for token in cleaned_tweet:
        new_sent = new_sent + lemmatizer.lemmatize(token) + ' '
    
    return new_sent.strip()


# In[ ]:


# Apply this cleaner function on ALL the tweets
# tweets["cleaned_tweets"] = tweets["tweet"].apply(tweet_cleaner)
# tweets.head()


# ## CREATE PIPELINE WHICH INCLUDES THE DATA CLEANER FUNCTION, ALONGWITH THE TF-IDF & LOGISTIC REGRESSION

# In[21]:


from sklearn.base import BaseEstimator, TransformerMixin

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [tweet_cleaner(text) for text in X]


# # Model Building using Pipeline

# In[22]:


TFIDF = TfidfVectorizer(max_df=0.5, min_df=10, max_features=300)
LR = LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2', C=0.5)

model_pipeline = Pipeline([('cleaner', TextCleanerTransformer()), \
                           ("TFIDF", TFIDF), ("LR", LR)])


# In[ ]:





# In[31]:


X = tweets['tweet']
y = tweets['label']
model_pipeline.fit(X,y)


# In[35]:


# Save the trained pipeline
import joblib
joblib.dump(model_pipeline, 'model_pipeline.pkl')


# In[36]:


model_pipeline = joblib.load('model_pipeline.pkl')


# In[37]:


model_pipeline.predict(X)


# In[ ]:





# In[ ]:




