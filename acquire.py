import pandas as pd
import numpy as np
import os
from requests import get
from bs4 import BeautifulSoup



def get_blogs(refresh= False):
    if not os.path.isfile('codeup_blogs.csv') or refresh:

        url = 'https://codeup/com/blog/'
        headers = {'User-Agent': 'Codup Data Science'}
        response = get(url, headers= headers)

        soup = BeautifulSoup(response.content, 'html.parser')
        links = [link['href'] for link in soup.select('h2 a[href]')]
        articles = []

        for url in links:
            url_response = get(url, headers= headers)
            soup = BeautifulSoup(url_response.text, 'html.parser')

            title= soup.find('h1', class_ = 'entry-title').text
            content = soup.find('div', class_ = 'entry-content').text.strip()

            article_dict = {'title': title, 'content': content}
            articles.append(article_dict)

        blog_article_df = pd.DataFrame(articles)
        blog_article_df.to_csv('codeup_blogs.csv')
    
    return pd.read_csv('codeup_blogs.csv')




def get_news(refresh= False):
    if not os.path.isfile('news.csv') or refresh:

        url = 'http://inshorts.com/en/read'
        response = get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        categories = [li.text.lower for li in soup.select('li')][1:]
        categories[0] = 'national'

        inshorts = []

        for category in categories:
            cat_url = url + '/' + category
            response = get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            titles = [span.text for span in soup.find_all('span', itemprop = 'headline')]
            contents = [div.text for div in soup.find_all('div', itemprop = 'articleBody')]

            for i in range(len(titles)):
                article = {'title': titles[i], 'content': contents[i], 'category': category}
                inshorts.append(article)
        
        inshorts_article_df = pd.DataFrame(inshorts)
        inshorts_article_df.to_csv('news_articles.csv', index = False)
    return pd.read_csv('news_articles.csv')






def get_blog_articles_data(refresh=False):
    
    if not os.path.isfile('blog_articles.csv') or refresh:
        
        url = 'https://codeup.com/blog/'
        headers = {'User-Agent': 'Codeup Data Science'}
        response = get(url, headers=headers)

        soup = BeautifulSoup(response.content, 'html.parser')

        links = [link['href'] for link in soup.select('h2 a[href]')]

        articles = []

        for url in links:

            url_response = get(url, headers=headers)
            soup = BeautifulSoup(url_response.text, 'html.parser')

            title = soup.find('h1', class_='entry-title').text
            content = soup.find('div', class_='entry-content').text.strip()

            article_dict = {
                'title': title,
                'content': content
            }

            articles.append(article_dict)
        
        blog_article_df = pd.DataFrame(articles)
        
        blog_article_df.to_csv('blog_articles.csv', index=False)
        
    return pd.read_csv('blog_articles.csv')

def get_news_articles_data(refresh=False):
    
    if not os.path.isfile('news_articles.csv') or refresh:
        
        url = 'https://inshorts.com/en/read'
        response = get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        categories = [li.text.lower() for li in soup.select('li')][1:]
        categories[0] = 'national'

        inshorts = []

        for category in categories:

            cat_url = url + '/' + category
            response = get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            titles = [span.text for span in soup.find_all('span', itemprop='headline')]
            contents = [div.text for div in soup.find_all('div', itemprop='articleBody')]

            for i in range(len(titles)):

                article = {
                    'title': titles[i],
                    'content': contents[i],
                    'category': category,
                }

                inshorts.append(article)
                
        inshorts_article_df = pd.DataFrame(inshorts)
        
        inshorts_article_df.to_csv('news_articles.csv', index=False)
                
    return pd.read_csv('news_articles.csv')