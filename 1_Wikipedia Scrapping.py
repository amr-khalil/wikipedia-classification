#!/usr/bin/env python
# coding: utf-8

# # Analyse semi- und unstrukturierter Daten

# # Aufgabe 2.1 | Wikipedia Scrapping
# 
# (praktische Aufgabe, Abgabe Code – mit kurzer Beschreibung der „wie“ in den ersten
# Kommentarzeilen, Inputdaten, Beispielsession mit Output)
# Bauen Sie den Wikipedia-Artikel-Kategorisierer so aus, dass drei verschiedene Artikelklasse
# voneinander unterschieden werden.

# # Load the txt files in a dictionary

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:35:53 2020

@author: Amr.Khalil
"""
# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os

# The website categories
categories = {'Kunst & Kultur':0, 'Sport':1, 'Wissenschaft':2}

# Create a dictionary for articles
titles = dict()
for category in categories.keys():
    with open('train/'+category+'.txt','r') as f:
        lis = f.read().split('\n')
        titles.update({category : [i.strip() for i in lis]})


# # Load all articles in one dictionary

# In[2]:


my_dict = dict()
# Every category must have a list
Kunst_und_Kultur = []
Sport = []
Wissenschaft = []

for category in categories.keys(): # Loop on the 3 categories
    for title in tqdm(titles[category], desc='{:<15}'.format(category)): # loop on the titels for every category
        url = "https://de.wikipedia.org/wiki/"+title # Parse the wikipedia page
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        paragraphs = soup.select('p') # select all the paragraphs in the article
        article_text = ""
        for para in paragraphs:
            article_text += para.text # Gather the paragraphs in one variable
        
        # Find the right category
        if category == 'Kunst & Kultur': 
            Kunst_und_Kultur.append(article_text)
            my_dict[category] = Kunst_und_Kultur
        
        elif category == 'Sport':
            Sport.append(article_text)
            my_dict[category] = Sport
        
        else:
            Wissenschaft.append(article_text)
            my_dict[category] = Wissenschaft
            


# # Create a Dataframe

# In[3]:


# Create a DataFrame for the first category
df1 = pd.DataFrame(my_dict['Kunst & Kultur'], columns =['Article'])
df1['Category'] = "Kunst & Kultur"
df1['Category_ID'] = 0

# Create a DataFrame for the second category
df2 = pd.DataFrame(my_dict['Sport'], columns =['Article'])
df2['Category'] = "Sport"
df2['Category_ID'] = 1

# Create a DataFrame for the third category
df3 = pd.DataFrame(my_dict['Wissenschaft'], columns =['Article'])
df3['Category'] = "Wissenschaft"
df3['Category_ID'] = 2

# collect them in one DataFrame
df = pd.concat([df1,df2,df3], ignore_index= True)
df[::30]


# In[4]:


# Numer of the articles for each category
df['Category'].value_counts()


# # Text processing

# In[5]:


import re
import nltk

# We need this dataset in order to use the tokenizer
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Also download the list of stopwords to filter out
#nltk.download('stopwords')
from nltk.corpus import stopwords

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
    
    #
    clean_text = [w for w in stemm_text if len(w) > 2]

    # This final output is a list of words
    
    return " ".join(clean_text)


# In[6]:


# Applying Data Processing on the DataFrame
# It will take sometime
df['Processed_Text'] = df['Article'].apply(process_text)


# # Save the Data into CSV Format

# In[7]:


df.to_csv('data.csv', index=False)

