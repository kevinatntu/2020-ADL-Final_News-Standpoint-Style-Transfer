from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# driver = webdriver.Firefox()
# driver.get("http://www.python.org")
# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()

import time
geckodriver = './geckodriver'
browser = webdriver.Firefox(executable_path=geckodriver)
extension_dir = '~/.mozilla/firefox/8mo4r7eq.default-release/extensions/'

extensions = ['{4283b371-3418-4240-8974-834c41786821}.xpi']
# print(extensions[0])
for extension in extensions:
    browser.install_addon(extension_dir+extension, temporary=True)

# browser.get('https://tw.appledaily.com/politics/20200606/HLGI4USDSW6IGFFTEAYHMRYMSA/')
browser.get('https://tw.appledaily.com/politics')
browser.find_element_by_id('search').send_keys('韓國瑜', Keys.RETURN)
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


time.sleep(5)

a=[]
a = browser.find_elements_by_class_name("rtbdg")
print(len(a))
# print(a)
# a[0].click()
# browser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 'w') 
# start = 0

texts = []
for i in range(len(a)):
    # a = browser.find_elements_by_class_name("rtbdg")
    a[i].click()
    texts.append(browser.find_element_by_tag_name('body').text)
    time.sleep(5)


    # browser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 'w') 
    # browser.close()
    # start += 1

# for l in a:
#     url = l.find_element_by_class_name("content")
#     url = url.find_element_by_tag_name('h2')
#     url = l.find_element_by_xpath("//*[@href]")
#     print (url.get_attribute('href'))
# links = browser.find_elements_by_xpath("//*[@item]//*[@href]")
# print(len(links))
# for link in links:
#     print (link.get_attribute('href'))

r = requests.get('https://tw.appledaily.com/politics/20200628/6LDHSNL47FOCS7HROADDJ7P7UY/')
soup = BeautifulSoup(r.text, 'html.parser')
# print(soup)
def get_target_urls(max_url = 300):
    url = 'https://tw.appledaily.com/politics/'
    # url = 'http://hk.racing.nextmedia.com/fullresult.php?date=20190123&page=01'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    options_text = str(soup.find_all('select')[0])

    date_urls_temp = re.findall('option value="(.*?)">', options_text)
    date_urls = [url for url in date_urls_temp if 'fullresult.php?date=' in url]

    all_target_urls = []
    for url in date_urls:
        r_date = requests.get('http://hk.racing.nextmedia.com/' + url)
        soup = BeautifulSoup(r_date.text, 'html.parser')
        urls_temp = [link.get('href') for link in soup.find_all('a')]
        all_target_urls += [u for u in urls_temp if 'fullresult.php?date' in u]
        print('len of url{}'.format(len(all_target_urls)))
        time.sleep(0.5)
        if len(all_target_urls) >= max_url:
            print('at least{}url'.format(max_url))
            return all_target_urls

def parse_url(url):
    failure_url = ''
    success = True
    url = 'http://hk.racing.nextmedia.com/' + url
    data = pd.read_html(url)
    game_id = re.findall('date=(.*?)&', url)[0]+'_'+re.findall('page=(.*?)$', url)[0]

    temp0 = data[1].iloc[1,0]
    env = temp0.split('\xa0')[0:5] + re.findall("([A-Z]\d?)",temp0) + re.findall("\((.*?)\)",temp0) + re.findall(":(.*?) ",temp0)
    env += re.findall(":(\d+)",temp0) + re.findall(":.*?(\d+.\d+)",temp0)
    if re.findall(":.*?(\d+.\d+)",temp0):
        env += re.findall(":.*?(\d+.\d+)",temp0)
    else:
        env += ['']
    env += re.findall(":.*?(\d+.\d+.\d+)",temp0)
    env.insert(0,game_id)
    

    if len(env) != 13:
        success = False
        failure_url = url
    


    
    detail = data[2].iloc[2:-2,:-1].reset_index(drop=True)
    

    
    try:
        detail.columns = columns
    except:
        success = False
        failure_url = url
        
        
    detail.insert(loc=0, column='game_id', value=[game_id]*len(detail))
    
    return env, detail, success, failure_url

if __name__ == '__main__':
    
    N = 300 
    all_urls = get_target_urls(max_url = N)
    
    envs = []
    details = []
    failed_urls = []

    
    
    for url in all_urls:
        env, detail, success, failure_url = parse_url(url)
        if success:
            envs.append(env)
            details.append(detail)
        else:
            failed_urls.append(failure_url)
            
        time.sleep(1)
    

    
    env_data = pd.DataFrame(envs)
    env_data.columns = env_column


