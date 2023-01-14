import os
import sys
from time import sleep, time
from tqdm import tqdm
import urllib.request
import requests as rq

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from multiprocessing import Pool
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures

from bs4 import BeautifulSoup

DATA_PATH = "../data/crawed_img/"

# reference : https://kimcoder.tistory.com/259
# brew install --cask chromedriver
# sudo mv chormedriver /usr/local/bin
    

def get_img_urls(keyword):
    options = webdriver.ChromeOptions()
    options.add_argument('headless') # 창 띄우지 않고 실행
    options.add_argument('--disable-gpu') # gpu 사용 x
    options.add_argument('lang=ko_KR') # 언어 한글
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")
    service = Service(executable_path="/usr/local/bin/chormedriver")

    img_urls = []
    with webdriver.Chrome(service=service, options=options) as browser:
        # 전체 페이지 갯수 구하기
        pixabay_url = f"https://pixabay.com/ko/images/search/{keyword}"
        browser.get(pixabay_url)
        sleep(.5)
        html = browser.page_source
        soup = BeautifulSoup(html,'html.parser')
        text = soup.find('form', 'add_search_params pure-form').getText()
        text = text.replace('/', '')
        start_idx = 0
        end_idx = 0
        for idx, t in enumerate(text):
            if t.isdigit() and start_idx == 0:
                start_idx = idx
            if start_idx != 0 and not t.isdigit():
                end_idx = idx
        try:
            pages = int(text[start_idx:end_idx])
        except:
            ValueError('ERROR(pages)')
        # img_urls 반환
        for page in tqdm(range(1, pages+1)):
            pixabay_url = f"https://pixabay.com/ko/images/search/{keyword}/?pagi={page}"
            browser.get(pixabay_url)
            sleep(.5)
            html = browser.page_source
            soup = BeautifulSoup(html,'html.parser')
            urls = soup.select('div.row-masonry.search-results img')
            for url in urls:
                img_urls.append(url)
    return img_urls

def get_img_and_save(img_url, keyword):
    srcset = ""
    if img_url.get('srcset') == None:
        srcset = img_url.get('data-lazy-srcset')
    else: 
        srcset = img_url.get('srcset')
        
    src = ""
    if len(srcset):
        src = str(srcset).split()[2] # 480 크기 이미지 가져옴 (기본 340)
        filename = src.split('/')[-1] #이미지 경로에서 날짜 부분뒤의 순 파일명만 추출
        saveUrl = os.path.join(DATA_PATH, keyword, filename) # 저장 경로 결정

        #파일 저장
        #user-agent 헤더를 가지고 있어야 접근 허용하는 사이트도 있을 수 있음(pixabay가 이에 해당)
        try:
            headers = {'User-Agent': 'Mozilla/5.0'} 
            timeout = 5
            img = rq.get(src, headers=headers, timeout=timeout)
            if img.status_code == 200:
                with open(saveUrl, "wb") as f: #디렉토리 오픈
                    f.write(img.content) #파일 저장
            else:
                print('status code error')
        except Exception as e:
            print(e)
            sys.exit(0)

def thread_crawling(keyword):
    thread_list = []
    img_urls = get_img_urls(keyword)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for img_url in img_urls:
            thread_list.append(executor.submit(get_img_and_save, img_url, keyword))
        for execution in concurrent.futures.as_completed(thread_list):
            execution.result()
            
def do_process_with_thread_crawl(keyword: str):
    thread_crawling(keyword)

    
if __name__ == "__main__":
    
    keywords = ['child', 'table', 'indoor']

    os.makedirs(DATA_PATH, exist_ok=True)
    for k in keywords:
        os.makedirs(os.path.join(DATA_PATH, k), exist_ok=True)
        
    start_time = time()
    
    with Pool(processes=3) as pool: # keyword마다 프로세스 할당
        pool.map(do_process_with_thread_crawl, keywords)

    end_time = time()
    
    print(f"소요시간: {end_time - start_time}")
    