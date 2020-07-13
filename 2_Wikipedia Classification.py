#!/usr/bin/env python
# coding: utf-8

# # Analyse semi- und unstrukturierter Daten

# ## Aufgabe 2.2 | Wikipedia Classification


# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:35:53 2020
@author: Amr.Khalil
"""
import requests
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('data.csv')
df[::30]


# In[3]:


# Split the Data into train and Test to measure the accuray
X = df.Processed_Text
y = df.Category_ID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state =41)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


# # Logistic Regression Classifier

# In[4]:


pipe = Pipeline([('vect', CountVectorizer(max_features=3000)),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

LR_clf = pipe.fit(X_train, y_train)
prediction = LR_clf.predict(X_test)
print("Logistic Regression Classifier Accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # Linear Support Vector Classifier

# In[5]:


pipe = Pipeline([('vect', CountVectorizer(max_features=3000)),
                 ('tfidf', TfidfTransformer()),
                 ('model', LinearSVC())])

LSVC_clf = pipe.fit(X_train, y_train)
prediction = LSVC_clf.predict(X_test)
print("Linear Support Vector Classifier Accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# # KNN Classifier

# In[6]:


pipe = Pipeline([('vect', CountVectorizer(max_features=3000)),
                 ('tfidf', TfidfTransformer()),
                 ('model', KNeighborsClassifier(n_neighbors=3))])

KNN = pipe.fit(X_train, y_train)
prediction = KNN.predict(X_test)
print("KNN Classifier Accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2))) # KNN is the best accuracy


# In[7]:


import requests
from bs4 import BeautifulSoup
import re
import nltk

# We need this dataset in order to use the tokenizer
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Also download the list of stopwords to filter out
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Import the snowball stemmer for german language
from nltk.stem.snowball import GermanStemmer
stemmer = GermanStemmer()

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-zäöüß]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    stemm_text = [stemmer.stem(w) for w in tokenized_text if w not in stopwords.words('german')]
    
    # Remove any word less then 2 charachters
    clean_text = [w for w in stemm_text if len(w) > 2]

    # This final output is a list of words
    return " ".join(clean_text)


# In[8]:


# Make prediction
def perdiction(titles):
    for title in titles:
        page = requests.get('https://de.wikipedia.org/wiki/'+str(title))
        soup = BeautifulSoup(page.content, 'html.parser')
        paragraphs = soup.select('p')
        article_text = ""
        for para in paragraphs:
            article_text += para.text

        clean_text = process_text(article_text)
        pred = KNN.predict([clean_text])

        id_to_category = {0:'Kunst & Kultur', 1:'Sport', 2:'Wissenschaft'}

        result = id_to_category[pred[0]]
        print("{} article is: {}".format(title, result)) 


# In[9]:


articles = ['FC_Bayern','Marie_Curie', 'Johann_Wolfgang_von_Goethe']
perdiction(articles)


# In[ ]:




