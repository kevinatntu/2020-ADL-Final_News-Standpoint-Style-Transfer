import requests
from requests_html import HTML
from pprint import pprint
import argparse
import os
import re

import numpy as np
import pandas as pd

# from multiprocessing import Pool7
import re
import os

def parse():
    parser = argparse.ArgumentParser(description="News Crawler")
    parser.add_argument('--platform', type=str, required=True, help='News platform')
    parser.add_argument('--keyword', type=str, default='', help='Searching keyword')
    parser.add_argument('--category', type=str, default='', help='Searching keyword')
    parser.add_argument('--pages', type=int, default=1, help='Number of parge you would like to search')
    parser.add_argument('--retry', action='store_true', help='Retry those failed entries')

    args = parser.parse_args()
    return args


class SimpleParser():
    '''
    Basic parser parent class
    '''
    def __init__(self, start_url):
        # setting the parser
        #self.domain = "https://www.ptt.cc"
        self.start_url = start_url
    
    def fetch(self, url):
        '''
        fetch the htmltext of corresponding url, with specific crawling format if needed
        '''
        response = requests.get(url)
        return response

    def parse_article_entries(self, doc):
        '''
        fetch every entry of current page
        '''
        pass

    def parse_article_meta(self, entry):
        '''
        parse current entry then return its metadata
        '''
        pass

    def parse_nextlink(self, doc):
        '''
        parse next link by htmltext
        '''
        pass

    def get_pages_meta(self, url, numPages=5):
        '''
        Given # of pages, return all entries' metadata of all pages
        '''
        pass
    
    def get_metadata(self, url):
        '''
        parse the corresponding htmltext
        '''
        pass

    def print_meta(self, meta):
        '''
        Print out the metadata
        '''
        pass

class ChinaTimesParser(SimpleParser):
    '''
    Simple parser for chinatimes.com (中時新聞網)
    '''
    def __init__(self, start_url, search_type='all'):
        super().__init__(start_url)
        self.domain = "https://www.chinatimes.com/Search/"
        self.columns = ['title', 'author', 'time', 'category', 'tag', 'intro', 'content', 'link']
    
    # def fetch(self, url): 
    # need not override it

    def parse_article_entries(self, doc):
        html = HTML(html=doc)
        post_entries = html.find("div.articlebox-compact")
        #post_entries_even = html.find("li.even")
        #post_entries_odd = html.find("li.odd")
        #return post_entries_even, post_entries_odd
        return post_entries

    def parse_article_contentpage_meta(self, content_page):
        html = HTML(html=content_page)
        post = html.find("div.article-wrapper")[0]
        
        author = post.find('div.author a', first=True)
        if author is None:
            author = post.find('div.author', first=True)
        author = author.text
        content_p = post.find("div.article-body p")
        content_p = [c.text for c in content_p if c.text != ""]
        content = '\n'.join(content_p)
        # tag - list
        tag = post.find("span.hash-tag a")
        tag = [t.text for t in tag]

        return author, content, tag

    def parse_article_meta(self, entry):
        '''
        Entry:
            'title':[], 'author':[], 'time':[], 'category':[], 'tag':[], 'intro':[], 'content':[], 'link':[]
        '''
        try:
            meta = {
                'title': entry.find("h3.title a", first=True).text,
                'author': "",
                'time': entry.find("div.meta-info time", first=True).attrs['datetime'],
                'category': entry.find("div.meta-info div.category a", first=True).text,
                'tag': "",
                'intro': entry.find("p.intro", first=True).text,
                'content': "",
                'link': entry.find("h3.title a", first=True).attrs['href']
            }

            # fetch content
            content_page = self.fetch(meta['link'])
            meta['author'], meta['content'], meta['tag'] = self.parse_article_contentpage_meta(content_page.text)
        except Exception:
            # ignore
            return None

        return meta
    
    def print_meta(self, meta):
        print("{}\n{} | {} | {} | {}\n{}\n{}\nLink: {}".format(meta['title'], meta['author'], meta['time'], meta['category'], meta['tag'], meta['intro'], 
            meta['content'], meta['link']))
            
        print()
        print("------------------------------")
        

class SETnParser(SimpleParser):
    '''
    Simple parser for www.setn.com/ (三立新聞網)
    '''
    def __init__(self, start_url, search_type='all'):
        super().__init__(start_url)
        self.domain = "https://www.setn.com/"
        self.search_type = search_type
        self.columns = ['title', 'author', 'time', 'category', 'tag', 'intro', 'content', 'link']
    
    # def fetch(self, url): 
    # need not override it

    def parse_article_entries(self, doc):
        html = HTML(html=doc)
        post_entries = html.find("div.newsimg-area-item-2 ")
        #post_entries_even = html.find("li.even")
        #post_entries_odd = html.find("li.odd")
        #return post_entries_even, post_entries_odd
        return post_entries

    def parse_article_contentpage_meta(self, content_page):
        html = HTML(html=content_page)
        post = html.find("div.page-text article")[0]
        
        author = post.find('div#Content1 p', first=True).text
        if author is None or not author:
            author = "Cannot find author"
        re1 = re.match('記者(.+)／', author)
        re2 = re.match('(.+)報導', author[author.rfind('／')+1:])
        re3 = re.match('(.+)／', author)
        if re1:
            author = re1.groups()[0]
        elif re2 and re2.groups()[0] != '綜合':
            author = re2.groups()[0]
        elif re3:
            author = re3.groups()[0]

        # TODO: remove middle image description word
        content_p = post.find("div#Content1 p[style!='text-align: center']")[1:]
        content_p = [c.text for c in content_p if c.text != "" and c.text[0] != '▲']
        content = ''.join(content_p)
        # tag - list
        post = html.find("div.page-keyword div.keyword")[0]
        tag = post.find("ul li a.gt strong")
        tag = [t.text for t in tag]

        # get full title
        full_title = html.find('h1.news-title-3')[0].text

        return full_title, author, content, tag

    def parse_article_meta(self, entry):
        '''
        Entry:
            'title':[], 'author':[], 'time':[], 'category':[], 'tag':[], 'intro':[], 'content':[], 'link':[]
        '''
        try:
            meta = {
                'title': entry.find("div.newsimg-area-text-2", first=True).text,
                'author': "",
                'time': entry.find("div.label-area div.newsimg-date", first=True).text,
                'category': entry.find("div.label-area div.newslabel-tab", first=True).text,
                'tag': "",
                'intro': entry.find("div.newsimg-area-info a.gt", first=True).text,
                'content': "",
                'link': entry.find("a.gt", first=True).attrs['href']
            }
            meta['link'] = prefix + meta['link'][:meta['link'].find("&From")]

            # fetch content
            content_page = self.fetch(meta['link'])
            meta['title'], meta['author'], meta['content'], meta['tag'] = self.parse_article_contentpage_meta(content_page.text)
        except Exception:
            # ignore
            return None

        return meta

    def print_meta(self, meta):
        print()
        print("{}\n{} | {} | {} | {}\n{}\n{}\nLink: {}".format(meta['title'], meta['author'], meta['time'], meta['category'], meta['tag'], meta['intro'], 
            meta['content'], meta['link']))
            
        print()
        print("------------------------------")


class China81(SimpleParser):
    '''
    Simple parser for http://www.81.cn/ (中國軍網)
    '''
    def __init__(self, start_url, search_type='all'):
        super().__init__(start_url)
        self.domain = "http://www.81.cn/"
        self.search_type = search_type
        self.columns = ['title', 'time', 'content', 'link']
    
    def fetch(self, url): 
        # use utf-8 encoding 
        response = requests.get(url)
        response.encoding = 'utf-8'
        return response

    def parse_article_entries(self, doc):
        html = HTML(html=doc)
        print(html.encoding)
        post_entries = html.find("ul#main-news-list li")
        #post_entries_even = html.find("li.even")
        #post_entries_odd = html.find("li.odd")
        #return post_entries_even, post_entries_odd
        return post_entries

    def parse_article_contentpage_meta(self, content_page):
        html = HTML(html=content_page)
        post = html.find("div.article-content")[0]

        content_p = post.find("p")
        content_p = [c.text for c in content_p]
        content = ''.join(content_p)

        return content

    def parse_article_meta(self, entry):
        '''
        Entry:
            'title':[], 'time':[], 'content':[], 'link':[]
        '''
        try:
            meta = {
                'title': entry.find("h3 span.title", first=True).text,
                'time': entry.find("h3 small.time", first=True).text,
                'content': "",
                'link': entry.find("a", first=True).attrs['href']
            }
            meta['link'] = prefix + meta['link']

            # fetch content
            content_page = self.fetch(meta['link'])
            meta['content'] = self.parse_article_contentpage_meta(content_page.text)
            #with open('./test.txt', 'w', encoding='big5') as ff:
            #    ff.write(meta['content'])
        except Exception:
            # ignore
            return None

        return meta

    def print_meta(self, meta):
        print()
        print("{}\n{}\n{}\nLink: {}".format(meta['title'], meta['time'], meta['content'], meta['link']))
            
        print()
        print("------------------------------")


class ChinaGWY(SimpleParser):
    '''
    Simple parser for http://big5.gwytb.gov.cn/ (國台辦)
    '''
    def __init__(self, start_url, search_type='all'):
        super().__init__(start_url)
        self.domain = "http://big5.gwytb.gov.cn/xwfbh/"
        self.search_type = search_type
        self.columns = ['title', 'time', 'content', 'link']
    
    def fetch(self, url): 
        # use utf-8 encoding 
        response = requests.get(url)
        response.encoding = 'big5hkscs'
        return response

    def parse_article_entries(self, doc):
        html = HTML(html=doc)
        print(html.encoding)
        post_entries = html.find("ul.black14pxlist li")
        #post_entries_even = html.find("li.even")
        #post_entries_odd = html.find("li.odd")
        #return post_entries_even, post_entries_odd
        return post_entries

    def parse_article_contentpage_meta(self, content_page):
        html = HTML(html=content_page)
        post = html.find("div.sharebox div#main div.TRS_PreAppend")[0]

        content_p = post.find("p.MsoNormal span")
        content_p = [c.text for c in content_p]
        content = ''.join(content_p)

        return content

    def parse_article_meta(self, entry):
        '''
        Entry:
            'title':[], 'time':[], 'content':[], 'link':[]
        '''
        try:
            meta = {
                'title': entry.find("a", first=True).text,
                'time': entry.find("a", first=True).text,
                'content': "",
                'link': entry.find("a", first=True).attrs['href']
            }
            title_end = meta['title'].find('（')
            meta['title'] = meta['title'][:title_end]
            meta['time'] = meta['time'][title_end+1:-1]

            meta['link'] = meta['link'] # no prefix

            # fetch content
            content_page = self.fetch(meta['link'])
            meta['content'] = self.parse_article_contentpage_meta(content_page.text)
            #with open('./test.txt', 'w', encoding='big5') as ff:
            #    ff.write(meta['content'])
        except Exception:
            # ignore
            return None

        return meta

    def print_meta(self, meta):
        print()
        print("{}\n{}\n{}\nLink: {}".format(meta['title'], meta['time'], meta['content'], meta['link']))
            
        print()
        print("------------------------------")


# calculate Chinese sentenses' true length in python print function
widths = [
    (126,  1), (159,  0), (687,   1), (710,  0), (711,  1),
    (727,  0), (733,  1), (879,   0), (1154, 1), (1161, 0),
    (4347,  1), (4447,  2), (7467,  1), (7521, 0), (8369, 1),
    (8426,  0), (9000,  1), (9002,  2), (11021, 1), (12350, 2),
    (12351, 1), (12438, 2), (12442,  0), (19893, 2), (19967, 1),
    (55203, 2), (63743, 1), (64106,  2), (65039, 1), (65059, 0),
    (65131, 2), (65279, 1), (65376,  2), (65500, 1), (65510, 2),
    (120831, 1), (262141, 2), (1114109, 1),
]

def get_width(s):
    """Return the screen column width for unicode ordinal o."""
    global widths
    #print(s)
    def char_width(o):
        if o == 0xe or o == 0xf:
            return 0
        for num, wid in widths:
            #print(o, num)
            if o <= num:
                return wid
        return 1
    return sum(char_width(ord(c)) for c in s)
    

if __name__ == '__main__':
    
    '''
    Command mode: 現在支援文字輸入搜尋了!
    '''

    args = parse()
    
    if args.platform == 'china':
        if args.keyword != '':
            start_url = "https://www.chinatimes.com/Search/[keyword]?page=[page]&chdtv"
        else:
            start_url = "https://www.chinatimes.com/Search/[keyword]?page=[page]&chdtv"
        newsParser = ChinaTimesParser(start_url)
    elif args.platform == 'setn':
        # r=0: 全文搜尋; r=1: 標題搜尋
        start_url = "https://www.setn.com/search.aspx?q=[keyword]&r=0&p=[page]"
        prefix = "https://www.setn.com/"
        newsParser = SETnParser(start_url)
    elif args.platform == '81cn':
        start_url = "http://www.81.cn/big5/xwfyr/node_105161[page].htm"
        prefix = "http://www.81.cn/big5/xwfyr/"
        newsParser = China81(start_url)
    elif args.platform == 'gwy':
        start_url = "http://big5.gwytb.gov.cn/xwfbh/index[page].htm"
        prefix = ""
        newsParser = ChinaGWY(start_url)

    else:
        print("Unsupported platform!")
        exit(1)
    
    print("Search {} in {}".format(args.keyword, args.platform))
    
    if not os.path.exists('./data/' + args.platform):
        os.mkdir('./data/' + args.platform)
    if not os.path.exists('./data/' + args.platform + '/' + args.keyword):
        os.mkdir('./data/' + args.platform + '/' + args.keyword)

    start_url = start_url.replace("[keyword]", args.keyword)
    print(start_url)

    pages = range(1, args.pages+1)
    if args.platform == 'china':
        print("共查找 {} 頁 (大約{} 筆新聞)".format(args.pages, args.pages * 20))
    elif args.platform == 'setn':
        print("共查找 {} 頁 (大約{} 筆新聞)".format(args.pages, args.pages * 36))

    #os.system("pause")

    #news_record = {'title':[], 'author':[], 'time':[], 'category':[], 'tag':[], 'intro':[], 'content':[], 'link':[]}
    news_record = []
    idx = 0
    failed = 0
    failed_record = []
    for page in pages:
        print("Parse page {} ...".format(page))

        if args.platform == '81cn':
            if page == 1:
                this_page_url = start_url.replace("[page]", '')
            elif page < 4:
                this_page_url = start_url.replace("[page]", '_{}'.format(page))
        elif args.platform == 'gwy':
            if page == 1:
                this_page_url = start_url.replace("[page]", '')
            else:
                this_page_url = start_url.replace("[page]", '_{}'.format(page-1))
        else:
            this_page_url = start_url.replace("[page]", str(page))
        print(this_page_url)

        #ctParser = ChinaTimesParser(this_page_url)
        metapage = newsParser.fetch(this_page_url)

        #print(metapage.text)
        #exit(0)

        #even_entries, odd_entries = ctParser.parse_article_entries(metapage.text)
        entries = newsParser.parse_article_entries(metapage.text)

        #print(len(even_entries), len(odd_entries))

        #entries = even_entries + odd_entries
        print(len(entries))

        if not entries:
            print("\n找不到此關鍵字之資料!\n")

        print("------------------------------")
        for entry in entries:
            '''
            if not ctParser.entry_filter(entry, option=option):
                continue
            print()
            '''
            idx += 1
            print("Parse {}...".format(idx), end='')

            meta = newsParser.parse_article_meta(entry)

            #newsParser.print_meta(meta)
            
            print("finish")
            if meta is not None:
                news_record.append(meta)
            else:
                failed += 1
                if args.retry:
                    failed_record.append([idx, entry])
        # Re-fetch those failed entries
        if args.retry and failed_record:
            for en_idx, entry in failed_record:
                print("Re-Parse {}...".format(en_idx), end='')

                meta = newsParser.parse_article_meta(entry)

                #newsParser.print_meta(meta)
                
                if meta is not None:
                    print("finish")
                    news_record.append(meta)
                else:
                    print("Still failed. Ignore.")

            failed_record = []

        print("Page {} finish".format(page))

        if page % 10 == 0:
            # save as csv
            news_record = pd.DataFrame(news_record, columns=newsParser.columns)
            if args.platform != '81cn':
                news_record.to_csv('./data/{}/{}/{}_{}_page_{}-{}.csv'.format(args.platform, args.keyword, args.platform, args.keyword, max(1, page-9), page), index=False, encoding='utf_8_sig')
            else:
                news_record.to_csv('./data/{}/{}_page_{}-{}.csv'.format(args.platform, args.platform, max(1, page-9), page), index=False, encoding='utf_8_sig')

            news_record = []

    print("Total: {}, {} failed".format(idx, failed))

    if page < 10:
        news_record = pd.DataFrame(news_record, columns=newsParser.columns)
        if args.platform != '81cn':
            news_record.to_csv('./data/{}/{}/{}_{}_page_{}-{}.csv'.format(args.platform, args.keyword, args.platform, args.keyword, max(1, page-9), page), index=False, encoding='utf_8_sig')
        else:
            news_record.to_csv('./data/{}/{}_page_{}-{}.csv'.format(args.platform, args.platform, max(1, page-9), page), index=False, encoding='utf_8_sig')

    

    