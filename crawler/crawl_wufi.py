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


# proxies = []
# def get_random_proxies():
#     # get 20 proxy each time
#     if proxies==[]:
#         response = requests.get('https://free-proxy-list.net/')
#         res = BeautifulSoup(response.text, features='html.parser')
#         ip_address_list = [element.text for element in res.select('td:nth-child(1)')[:20]]
#         port_list = [element.text for element in res.select('td:nth-child(2)')[:20]]
#         for ip, port in zip(ip_address_list, port_list): 
#             proxies.extend([str(ip)+':'+str(port)])
#     random_prox = choice(proxies)
#     proxies.remove(random_prox)
#     return random_prox


# random_prox = get_random_proxies()
# print('current proxy: ' + random_prox)
# proxy = {
#     "http": ('http://' if 'http://' not in random_prox else '') + random_prox, 
#     "https": ('http://' if 'http://' not in random_prox else '') + random_prox, 
# }

requests.adapters.DEFAULT_RETRIES = 3   # reload times if fail
session = requests.Session()  
while True: 
    try: 
        session.get('https://www.google.com.tw/', allow_redirects=False)
        print('connected to Google successfully')
        break
    except Exception as e:
        print(e)
        # random_prox = get_random_proxies()  # change proxy 
        # print('Change proxy to '+random_prox)
        # proxy = {
        #     "http": ('http://' if 'http://' not in random_prox else '') + random_prox, 
        #     "https": ('http://' if 'http://' not in random_prox else '') + random_prox, 
        # }
        # continue

def get_url_text(url):
    global proxy
    while 1:            
        try:
            session.keep_alive = False
            session.headers = {'user-agent':str(UserAgent().random), 'accept-language':'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5'}
            res = session.get(url, allow_redirects=True)
            res.encoding = 'utf-8'
            return res.text
        except Exception as e:
            print(e)
            # print('request error')
            # random_prox = get_random_proxies()  # change proxy 
            # print('Change proxy to '+random_prox)
            # proxy = {
            #     "http": ('http://' if 'http://' not in random_prox else '') + random_prox, 
            #     "https": ('http://' if 'http://' not in random_prox else '') + random_prox, 
            # }
            # continue


def wufi_reader(url):
    text = get_url_text(url)
    res = BeautifulSoup(text, features='html.parser')

    title = res.find('h1', attrs={'class':'post-title item fn'}).text.strip()

    context = ""
    P = res.findAll('div', {'itemprop':'articleBody'})
    for p in P[0].findAll('p'):
        if len(p.text.strip()) != 0:
            paragraph = (p.text.strip('\n').strip())
            if '【台獨聯盟聲明稿】' not in paragraph and '【台灣獨立建國聯盟聲明稿】' not in paragraph:
                context += paragraph + '。'   
    
    for i in range(len(context)):
        if context[i] == ',': 
            context = context[:i] + '，' + context[i+1:]


    news_dict = {'title':title, 'author':None, 'context':context, 'url':url, 'tag':None, 'time':None, 'related_news':None,'source_video':None, 'source_img':None, 'media':'台獨聯盟'}       
    return news_dict


def crawl_by_page(page=1):
    url = 'https://www.wufi.org.tw/category/editorial/' + ('' if page==1 else f'page/{page}/')
    text = get_url_text(url)

    if 'Page Not Found' in text:
        return []

    res = BeautifulSoup(text, features='html.parser')
    max_page = int(res.findAll('a',{'class':'page-numbers'})[-2].text)
    drinks = res.findAll('h2',{'itemprop':'name'})
    drinks = [d.findAll('a')[0].get('href') for d in drinks]

    all_news = []
    for drink in drinks:
        news_url = drink
        try:
            news_dict = wufi_reader(news_url)
            all_news.append(news_dict)    
            time.sleep(3)
        except Exception as e: 
            print(e)
            continue    
    if page == 1:
        return all_news, max_page
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
    max_page = '???'
    while True:
        print(f'crawling page {page}/{max_page}', end='...')
        if page == 1:
            news, max_page = crawl_by_page(page=page)
        else:
            news = crawl_by_page(page=page)
        if len(news) == 0:
            print('None')
            break
        else:
            print(len(news))
        page += 1
        all_news += news
        

    if len(all_news) > 0:
        pack_to_df(all_news).to_csv('wufi.csv', index=False, encoding='utf-8')



if __name__ == "__main__":
    crawl()