# -*- coding: utf8 -*- 
import requests
import re # string resplit
from bs4 import BeautifulSoup # crawler main func
import ast # string to dict
import pandas as pd # sort dict
import json, os # input, output json file # check file exist
import sys # stop as read last progress
from urllib.parse import quote

# avoid crawler error
from time import sleep # sleep as be block for over-visiting
from fake_useragent import UserAgent # random user agent
from random import choice # random choice
import socket

# shedule execute span
from datetime import datetime, date, timedelta
import time

import pandas as pd


requests.adapters.DEFAULT_RETRIES = 3   # reload times if fail
session = requests.Session()  
while True: 
    try: 
        session.get('https://www.google.com.tw/', allow_redirects=False)
        print('connected to Google successfully')
        break
    except Exception as e:
        print(e)


def get_url_text(url):
    global proxy
    while 1:            
        try:
            session.keep_alive = False
            session.headers = {'user-agent':str(UserAgent().random), 'accept-language':'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5'}
            res = session.get(url, allow_redirects=True)
            res.encoding = 'big5'
            return res.text
        except Exception as e:
            print(e)


def winnie_reader(url):
    text = get_url_text(url)
    res = BeautifulSoup(text, features='html.parser')

    title = res.find('h1').text.strip()

    if '馬曉光' in res.text:
        context = ""
        P = res.find('div', {'id':'main'}).findAll('p')
        for p in P:
            if '馬曉光：' in p.text:
                idx = p.text.find('馬曉光：') + len('馬曉光：')
                paragraph = (p.text[idx:].strip('\n').strip())
                context += paragraph + '。'

    else:
        context = ""
        P = res.find('div', {'id':'main'}).findAll('p')
        for p in P:
            if len(p.text.strip()) != 0:
                paragraph = (p.text.strip('\n').strip())
                context += paragraph + '。'
    
    for i in range(len(context)):
        if context[i] == ',': 
            context = context[:i] + '，' + context[i+1:]


    news_dict = {'title':title, 'author':None, 'context':context, 'url':url, 'tag':None, 'time':None, 'related_news':None,'source_video':None, 'source_img':None, 'media':'台獨聯盟'}       
    return news_dict


def crawl_by_page(page=0):
    url = 'http://big5.gwytb.gov.cn/xwfbh/' + ('' if page==0 else f'index_{page}.htm')
    text = get_url_text(url)

    if '404 Not Found' in text:
        return []

    res = BeautifulSoup(text, features='html.parser')
    drinks = res.findAll('ul',{'class':'black14pxlist'})[0].findAll('li')
    drinks = [d.findAll('a')[0].get('href') for d in drinks]

    all_news = []
    for drink in drinks:
        news_url = drink
        try:
            news_dict = winnie_reader(news_url)
            all_news.append(news_dict)    
            time.sleep(3)
        except Exception as e: 
            print(e)
            continue    
    return all_news



def pack_to_df(news):
    d = {'title':[], 'author':[], 'time':[], 'category':[], 'tag':[], 'intro':[], 'content':[], 'link':[]}

    for n in news:
        d['title'].append(n['title'])
        d['author'].append(None)
        d['time'].append(None)
        d['category'].append(None)
        d['tag'].append(None)
        d['intro'].append(None)
        d['content'].append(n['context'])
        d['link'].append(n['url'])

    return pd.DataFrame.from_dict(d)


def crawl():
    page = 1

    all_news = []
    for page in range(7):
        print(f'crawling page {page+1}/7', end='...')
        news = crawl_by_page(page=page)
        if len(news) == 0:
            print('None')
            break
        else:
            print(len(news))
        page += 1
        all_news += news
        

    if len(all_news) > 0:
        pack_to_df(all_news).to_csv('gwytb.csv', index=False, encoding='utf-8')



if __name__ == "__main__":
    crawl()