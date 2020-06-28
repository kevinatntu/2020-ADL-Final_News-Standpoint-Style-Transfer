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


def ltn_reader(url):
    text = get_url_text(url)
    res = BeautifulSoup(text, features='html.parser')

    title = res.find('meta', attrs={'property':'og:title'})['content'].split(u' - ')[0]
    tag_list = res.find('meta', attrs={'name':'keywords'})['content'].split(u',')

    # reported time: {time, time_year, time_month, time_day, time_hour_min}
    if res.select('.whitecon .viewtime') != []:
        time_str = res.find('meta', attrs={'name':'pubdate'})['content'].strip('\n').strip()
        try:  date_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ') # ex. 2019-09-20T05:30:00Z
        except Exception as e: date_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') # ex. 2019-09-20T05:30:00
    elif res.select('.time') != []:
        time_str = res.select('.time')[0].text.strip('\n').strip()
        try:  date_obj = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
        except Exception as e: date_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') # ex. 2019-09-20T05:30:00
    elif res.select('.date') != []:
        time_str = res.select('.date')[0].text.strip('\n').strip()
        try:  date_obj = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
        except Exception as e: date_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') # ex. 2019-09-20T05:30:00
    time = date_obj.strftime("%Y/%m/%d %H:%M")
    time_detail = [{'time':time, 'time_year':date_obj.strftime("%Y"), 'time_month':date_obj.strftime("%m"), 'time_day':date_obj.strftime("%d"), 'time_hour_min':date_obj.strftime("%H:%M")}]

    context = ""
    for p in res.select('.text > p') or res.select('.news_content > p'):
        if len(p.text) !=0 and p.text.find(u'想看更多新聞嗎？現在用APP看新聞還可以抽獎')==-1 and p.text.find(u'／特稿')==-1 and p.select('.ph_i')==[]:
            paragraph = (p.text.strip('\n').strip())
            context += paragraph
    if len(context.split(u'不用抽'))>1: context = context.split(u'不用抽 不用搶')[0]

    author = context.split(u'〔')[1].split(u'／')[0] if len(context.split(u'〔'))>1 else ''
    if author[:2] == '記者':
        author = author[2:]

    if re.match(".*〔.*／.*〕.*", context):
        context = context.split('〕')[1]
    elif re.match(".*［.*／.*］.*", context):
        context = context.split('］')[1]


    related_title = [element.get_text() for element in res.select('.whitecon .related a') if element.get('href').find('http')==0] # 相關新聞 標題
    related_url = [element.get('href') for element in res.select('.whitecon .related a') if element.get('href').find('http')==0] # 相關新聞 url
    related_news = [dict([element]) for element in zip(related_title, related_url)]

    # video: {video_title:video_url}
    if res.select('#art_video') == []: video = []
    else:
        video_title = [element.get('alt') for element in res.select('#art_video')]
        video_url = ['https://youtu.be/'+element.get('data-ytid') for element in res.select('#art_video')]
        video = [dict([element]) for element in zip(video_title, video_url)]    

    # img: {img_title:img_url}
    if res.select('.lightbox img, .articlebody .boxInput, .photo_bg img')==[]: img = [] # no img
    else:
        img_title = [element.get('alt') for element in res.select('.lightbox img, .articlebody .boxInput, .photo_bg img')]
        img_url = [element.get('src') for element in res.select('.lightbox img, .articlebody .boxInput, .photo_bg img')]
        img = [dict([element]) for element in zip(img_title, img_url)]          

    news_dict = {'title':title, 'author':author, 'context':context, 'url':url, 'tag':tag_list, 'time':time_detail, 'related_news':related_news,'source_video':video, 'source_img':img, 'media':'自由時報'}       
    return news_dict


def crawl_ltn_by_query(keyword, conditions='or', start_time='2020-03-14', end_time='2020-06-07', page=1):
    """
    args: 
        keyword: Iterable, up to 3 items. Extra keywords will be trimmed. 
    """

    keyword = '+'.join([quote(k) for k in keyword[:3]])
    url = f'https://news.ltn.com.tw/search?keyword={keyword}&conditions={conditions}&start_time={start_time}&end_time={end_time}' + ('' if page==1 else f'&page={page}')
    text = get_url_text(url)

    if '查無新聞！！' in text:
        return []

    res = BeautifulSoup(text, features='html.parser')
    drinks = res.findAll('a',{'class':'tit'})
    drinks = [d.get('href') for d in drinks]

    all_news = []
    for drink in drinks:
        news_url = drink
        try:
            news_dict = ltn_reader(news_url)
            all_news.append(news_dict)    
            time.sleep(5)
        except Exception as e: continue    
    return all_news 



def pack_to_df(news):
    d = {'title':[], 'author':[], 'time':[], 'category':[], 'tag':[], 'intro':[], 'content':[], 'link':[]}

    for n in news:
        d['title'].append(n['title'])
        d['author'].append(n['author'])
        d['time'].append(n['time'][0]['time'])
        d['category'].append('')
        d['tag'].append(n['tag'])
        d['intro'].append('')
        d['content'].append(n['context'])
        d['link'].append(n['url'])

    return pd.DataFrame.from_dict(d)


def crawl_ltn_by_keyword(keyword):
    """
    args: 
        keyword: Iterable, up to 3 items. Extra keywords will be trimmed. 
    """


    # page = 1
    # all_news = []
    # while True:
    #     print(f'crawling {keyword} conditions=or, start_time=2019-12-08, end_time=2020-01-07, page={page}', end='...')
    #     news = crawl_ltn_by_query(keyword, conditions='or', start_time='2019-12-08', end_time='2020-01-07', page=page)
    #     if len(news) == 0:
    #         print('None')
    #         break
    #     else:
    #         print(len(news))
    #     page += 1
    #     all_news += news

    # if len(all_news) > 0:
    #     pack_to_df(all_news).to_csv('2019_12_08_to_2020_01_07.csv', index=False)

    # for year, mrange in [('2020', range(5, 0, -2)), ('2019', range(11, 0, -2))]:
    for year, mrange in [('2019', range(7, 0, -2)), ]:
        for month in mrange:
            
            end_month = str(min(12, month+2)).zfill(2)
            start_month = str(month).zfill(2)
            page = 1

            all_news = []
            while True:
                print(f'crawling {keyword} conditions=or, start_time={year}-{start_month}-08, end_time={year}-{end_month}-07, page={page}', end='...')
                news = crawl_ltn_by_query(keyword, conditions='or', start_time=f'{year}-{start_month}-08', end_time=f'{year}-{end_month}-07', page=page)
                if len(news) == 0:
                    print('None')
                    break
                else:
                    print(len(news))
                page += 1
                all_news += news

            if len(all_news) > 0:
                pack_to_df(all_news).to_csv(f'{year}_{start_month}_08_to_{year}_{end_month}_07.csv', index=False)



if __name__ == "__main__":
    crawl_ltn_by_keyword(['韓國瑜'])