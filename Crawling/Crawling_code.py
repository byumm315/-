# -*- coding: utf-8 -*-
"""신문기사 크롤링

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KVX87201riQeyadQCqzOvYx3Jt8kXzZH
"""

# 10개의 신문사 중 경향신문의 방역 관련 기사 추출한 사례를 보여드리겠습니다.[검색어: 코로나 바이러스 방역정책]
import requests
from bs4 import BeautifulSoup
link=[]
for num in range(1,12):
  webpage = requests.get("https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%BD%94%EB%A1%9C%EB%82%98%20%EB%B0%94%EC%9D%B4%EB%9F%AC%EC%8A%A4%20%EB%B0%A9%EC%97%AD%EC%A0%95%EC%B1%85&sort=0&photo=3&field=0&pd=3&ds=2020.01.20&de=2020.10.31&cluster_rank=20&mynews=1&office_type=1&office_section_code=1&news_office_checked=1032&nso=so:r,p:from20200120to20201031,a:all&start="+str(10*(num-1)+1))
  soup = BeautifulSoup(webpage.content, "html.parser")
  for j in soup.find_all(attrs={'class':'news_tit'}):
    link.append(j['href'])

!pip install datefinder
import urllib.request
from urllib.request import urlopen
import pandas as pd
import datefinder
link_text=[[]]
temp=[]
header01={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
df= pd.DataFrame(index=range(0,len(link)), columns=['text', 'url','date'])
for obs in range(0,len(link)):
    link_text=[]
    url=str(link[obs])
    modi01=urllib.request.Request(url,headers=header01)
    html = urllib.request.urlopen(modi01).read()
    soup = BeautifulSoup(html, 'html.parser')
    link_text=[]
    date_1=[]
    url_1=[]
    dl=soup.find_all('div',{'class':'art_body'})
    da=soup.find_all('div',{'class':'byline'})
    if len(dl)>0:
        input_string=da[0].text
        matches=list(datefinder.find_dates(input_string))
        date=str(matches[0])[:10]
        for dd in dl[0].find_all('p',{'class':'content_text'}):
                    link_text.append(dd.text)
        tt=link_text[0]
        for i in range(1,len(link_text)):
            tt+=link_text[i]
        df.iloc[obs,:] = pd.Series({'text':tt,'url':url,'date':date}) #df로 변환     
    else:
        dl=soup.find_all('p',{'class':'art_text'})
        da=soup.find_all('div',{'class':'art_date'})
        if len(dl)>0:
          for dd in dl:
                      link_text.append(dd.text)
          tt=link_text[0]
          for i in range(1,len(link_text)):
              tt+=link_text[i]
          df.iloc[obs,:] = pd.Series({'text':tt,'url':url,'date':date}) #df로 변환     
        else:
          df.iloc[obs,:]=pd.Series({'text':'None','url':url,'date':'None'})
df.to_excel("방역_경향신문_기사추출.xlsx",sheet_name='sheet1',index=False)

df
