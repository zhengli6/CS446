__author__ = 'ZhengLi'
print(' WEB CRAWLER for Clash Royale match records')
import os
import requests
from bs4 import BeautifulSoup

def get_Matches(url):
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text,"html.parser")
        body = soup.body
        containers = body.find_all('div','container')
        container = containers[1]
        search = container.find_all('div',"panel panel-inverse")
        panel_inverse = search[1]
        search = panel_inverse.find_all('div',"panel-body")
        panel_body = search[0]
        search = panel_body.find_all('div',"panel panel-inverse")
        # match = search[0]
        print(panel_inverse.prettify())
        print(len(search))
        # while i < len(link):
        #     #print(i)
        #     if i%2 != 0:
        #         vs_t = link[i]
        #         x = BeautifulSoup(str(vs_t))
        #         sale_check = x.find('del')
        #         vs_text = vs_t.text
        #         split_text = vs_text.split()
        #         full_text = ' '.join(str(e) for e in split_text)
        #         name_t = full_text.split('$')

        #         if sale_check is None:
        #             string = name_t[1]
        #             price = string.split()
        #             price = price[0]
        #             name = name_t[0] + '[$' + price + ']'
        #         else:
        #             sale_price = x.find('p')
        #             price = sale_price.text
        #             price = price.split()
        #             name = 'ON SALE ' + price[1] + ' ' + name_t[0] + '[' + price[0] + ']'

        #         print(name)

        #         vs_u = str(link[i-1])
        #         soup2 = BeautifulSoup(vs_u)
        #         full_url = soup2.find('img')
        #         image_url = 'http:'+full_url.get('data-lazy-asset')

        #         download_web_image(image_url, name)

        #     i += 1
        # print('Downloading complete')
get_Matches(r'https://statsroyale.com/profile/PCQUCPJ0')