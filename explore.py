#Imports
import re
import unicodedata
import pandas as pd
import nltk

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud



#Define function to clean up text data
def clean(text, ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']):
    
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]



def show_counts_and_ratios(df, column):
    """
    takes in df and single column string name returns 
    count and precentage that it shows up
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    labels
    return labels



def create_bigrams(text):
    
    tokens = nltk.word_tokenize(text.lower())
    bigrams = list(nltk.bigrams(tokens))
    return bigrams




def create_wordcloud(text):
  
    wc = WordCloud(background_color="white", max_words=200, contour_width=3, contour_color='steelblue')
    wc.generate(text)
    
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()





