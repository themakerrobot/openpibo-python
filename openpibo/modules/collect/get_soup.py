from bs4 import BeautifulSoup
import requests

def get_soup(url, parser='html.parser'):
    headers = {'User-Agent' : 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content, parser)

if __name__ == "__main__":
    search_text = '사과'
    html = get_soup(f'https://www.coupang.com/np/search?q={search_text}&channel=user')
    print(html)
