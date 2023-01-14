import os
import requests as rq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup
from time import sleep
import urllib.request
import sys
from tqdm import tqdm

# reference : https://kimcoder.tistory.com/259
# brew install --cask chromedriver
# sudo mv chormedriver /usr/local/bin

DATA_PATH = "./crawed_img/"

keywords = ['indoor', 'child', 'table']

os.makedirs(DATA_PATH, exist_ok=True)
for k in keywords:
    os.makedirs(os.path.join(DATA_PATH, k), exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument('headless') # 창 띄우지 않고 실행
options.add_argument('--disable-gpu') # gpu 사용 x
options.add_argument('lang=ko_KR') # 언어 한글
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
options.add_argument(f"user-agent={user_agent}")
service = Service(executable_path="/usr/local/bin/chormedriver")

with webdriver.Chrome(service=service, options=options) as browser:
    count = 0
    for keyword in keywords:
        pixabay_url = f"https://pixabay.com/ko/images/search/{keyword}"
        browser.get(pixabay_url)
        sleep(.1)
        html = browser.page_source
        soup = BeautifulSoup(html,'html.parser')
        text = soup.find('form', 'add_search_params pure-form').getText()
        try:
            pages = int(text[5:7])
        except:
            ValueError(f'ERROR(pages : {pages})')
            
        for page in tqdm(range(1, pages+1)):
            pixabay_url = f"https://pixabay.com/ko/images/search/{keyword}/?pagi={page}"
            browser.get(pixabay_url)
            sleep(.1)
            html = browser.page_source
            soup = BeautifulSoup(html,'html.parser')
            imgs = soup.select('div.row-masonry.search-results img')
            
            for img in imgs:
                count += 1
                srcset = ""
                if img.get('srcset') == None:
                    srcset = img.get('data-lazy-srcset')
                else: 
                    srcset = img.get('srcset')
                    
                src = ""
                if len(srcset):
                    src = str(srcset).split()[2] # 480 크기 이미지 가져옴 (기본 340)
                    filename = src.split('/')[-1] #이미지 경로에서 날짜 부분뒤의 순 파일명만 추출
                    saveUrl = os.path.join(DATA_PATH, keyword, filename) #저장 경로 결정

                    #파일 저장
                    #user-agent 헤더를 가지고 있어야 접근 허용하는 사이트도 있을 수 있음(pixabay가 이에 해당)
                    req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
                    try:
                        imgUrl = urllib.request.urlopen(req).read() #웹 페이지 상의 이미지를 불러옴
                        with open(saveUrl, "wb") as f: #디렉토리 오픈
                            f.write(imgUrl) #파일 저장
                    except urllib.error.HTTPError:
                        print('에러')
                        sys.exit(0)

print(f"전체 이미지 갯수:{count}")
print("FIN")