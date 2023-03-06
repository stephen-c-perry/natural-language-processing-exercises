#unicode, regex, json for text digestion
import unicodedata
import re
import json
import nltk

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import pandas as pd
from time import strftime
from requests import get
from bs4 import BeautifulSoup

#ignore warnings import
import warnings
warnings.filterwarnings('ignore')


def basic_clean(string):

    ''' 
    Takes in string, makes it all lowercase, encoded in ascii all characters not ascii ignored.  
    decoded in utf-8 all characters not utf-8 ignored.  remove special characters.  return cleaned string
    '''
  
    string = string.lower()
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    
    return string



def tokenize(string):
    
    '''
    Takes in string and returns it as individual tokens put back into the string
    '''

    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)

    return string



def stem(string):

    ''' 
    takes in string, returns stem word joined back into the string
    '''

    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)

    return string



def lemmatize(string):

    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)

    return string



def remove_stopwords(string, extra_words = [], exclude_words = []):

    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    string = ' '.join(filtered_words)
    
    return string






def prep_data(df, column, extra_words=[], exclude_words=[]):

    '''
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''

    df['original'] = df['content']
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    df['stemmed'] = df['clean'].apply(stem)
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['title', 'original', 'clean', 'stemmed', 'lemmatized']]


