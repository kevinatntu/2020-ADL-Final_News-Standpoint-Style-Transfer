# News Crawler tool

## newsCrawler.py 
- Supported Datasets: 三立 (SETN), 中時 (Chinatimes), 中國軍網 (81cn)
- Command:
  
    > python newsCrawler.py --platform=[Platform name] --keyword=[Search keyword] --pages=[Max pages]
- Option: 
    - platform: china, setn, 81cn
    - keywords: keywords you want to search, e.g. 韓國瑜, 台獨
    - pages: number of max pages the tool will search

## crawl_*.py

* Supported Datasets: 自由時報 (LTN), 台獨聯盟(WUFI), 國台辦 (GWYTB)

* 自由時報 (LTN):

  ```
  python crawl_ltn.py
  ```

* 台獨聯盟(WUFI): 

  ```
  python crawl_wufi.py
  ```

* 國台辦 (GWYTB):

  ```
  python crawl_gwytb.py
  ```
  