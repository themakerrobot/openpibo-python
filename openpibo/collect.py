"""
인터넷에서 유용한 정보를 가져옵니다.

**단어정보, 날씨 정보, 뉴스 정보** 를 가져올 수 있습니다.

Class:
:obj:`~openpibo.collect.Wikipedia`
:obj:`~openpibo.collect.Weather`
:obj:`~openpibo.collect.News`
"""

from urllib.parse import quote
from bs4 import BeautifulSoup
import requests

class Wikipedia:
    """
Functions:
:meth:`~openpibo.collect.Wikipedia.search`

    위키백과에서 단어를 검색합니다.

    example::

        from openpibo.collect import Wikipedia

        pibo_wiki = Wikipedia()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
    """
    def search(self, search_text: str):
        """
        위키백과에서 ``search_text`` 를 검색합니다.

        example::

            result = pibo_wiki.search('강아지')

        :param str search_text: 위키백과에서의 검색어

        :returns: 내용을 dictionary 배열 형태로 반환합니다.

            대부분의 경우 '0'번 항목에 개요를 표시하고, 검색된 내용이 없을 경우 None을 반환합니다.

            example::

                ['0':{
                    'title': '명칭', 
                    'content': "한국어 ‘강아지’는 ‘개’에 어린 짐승을 뜻하는 ‘아지’가 붙은 말이다..."
                }, ... ]
                or
                None
        """

        _chapters = {'0':{'title':'개요', 'content':[]}}
        encode_text = quote(search_text)
        url = f'https://ko.wikipedia.org/wiki/{encode_text}'
        resp = requests.get(url, headers={'User-Agent' : 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.content, 'html.parser')
        total_content = soup.find('div', {'class': 'mw-parser-output'})

        if not total_content:
            return None

        # chapter_idx = '3.1', chapter_list = [3, 1]
        chapter_idx, chapter_list = '0', [0]
        parent_num = 0
        for content in total_content:
            tag = content.name
            if tag == None:
                continue
            elif tag[0] == 'h':
                new_parent_num = int(tag[1]) - 2
                if new_parent_num <= parent_num:
                    chapter_list = chapter_list[:new_parent_num+1]
                    chapter_list[-1] += 1
                else:
                    chapter_list.append(1)
                chapter_idx = '.'.join(map(str, chapter_list))
                _chapters[chapter_idx] = {}
                _chapters[chapter_idx]['title'] = content.text.split('[')[0]
                _chapters[chapter_idx]['content'] = []
            elif tag == 'p':
                _chapters[chapter_idx]['content'].append(content.text)
            elif tag == 'ul':
                _chapters[chapter_idx]['content'].append(content.text)

        return _chapters


class Weather:
    """
Functions:
:meth:`~openpibo.collect.Weather.search`

    종합 예보와 오늘/내일/모레의 날씨 정보를 검색합니다.

    example::

        from openpibo.collect import Weather

        pibo_weather = Weather()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
    """
    region_list = {
        '전국': 108,
        '서울': 109,
        '인천': 109,
        '경기': 109,
        '부산': 159,
        '울산': 159,
        '경남': 159,
        '대구': 143,
        '경북': 143,
        '광주': 156,
        '전남': 156,
        '전북': 146,
        '대전': 133,
        '세종': 133,
        '충남': 133,
        '충북': 131,
        '강원': 105,
        '제주': 184,
    }
    """
    날씨 정보를 검색할 수 있는 지역입니다.
    """

    def search(self, search_region:str='전국'):
        """
        해당 지역(```search_region```)의 날씨 정보(종합예보, 오늘/내일/모레 날씨)를 가져옵니다.

        example::

            result = pibo_weather.search('서울')

        :param str search_region: 검색 가능한 지역 (default: 전국)

            검색할 수 있는 지역은 다음과 같습니다::

                '전국', '서울', '인천', '경기', '부산', '울산', '경남', '대구', '경북',
                '광주', '전남', '전북', '대전', '세종', '충남', '충북', '강원', '제주'

        :returns: 종합예보와 오늘/내일/모레의 날씨 및 최저/최고기온을 반환합니다.

            example::

                {
                    'forecast': '내일 경기남부 가끔 비, 내일까지 바람 약간 강, 낮과 밤의 기온차 큼'
                    'today':
                    {
                        'weather': '전국 대체로 흐림',
                        'minimum_temp': '15.3 ~ 21.6',
                        'highst_temp': '23.1 ~ 27.6'
                    }
                    'tomorrow':
                    {
                        'weather': '전국 대체로 흐림',
                        'minimum_temp': '15.3 ~ 21.6', 
                        'highst_temp': '23.1 ~ 27.6'
                    }
                    'after_tomorrow':
                    {
                        'weather': '전국 대체로 흐림',
                        'minimum_temp': '15.3 ~ 21.6',
                        'highst_temp': '23.1 ~ 27.6'
                    }
                }
                or None
        """

        region = Weather.region_list.get(search_region)
        if region == None:
          raise Exception(f'"{search_region}" not support')

        _forecast = ''
        _today = {}
        _tomorrow = {}
        _after_tomorrow = {}

        url = f'https://www.weather.go.kr/w/weather/forecast/short-term.do?stnId={region}'
        resp = requests.get(url, headers={'User-Agent' : 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.content, 'html.parser')
        forecasts = soup.find('div', {'class': 'cmp-view-content'}).text
        forecasts = forecasts.split('□')[1].split('○')
        for forecast in map(str.strip, forecasts):
            split_point = forecast.index(')')
            date = forecast[:split_point+1]
            desc = forecast[split_point+2:]
            if '종합' in date:
                _forecast = desc
            if '오늘' in date:
                _today['weather'] = desc
            if '내일' in date or '~' in date:
                _tomorrow['weather'] = desc
            if '모레' in date:
                _after_tomorrow['weather'] = desc

        temp_table = soup.find('tbody')
        all_temps = list(map(lambda x: x.text, temp_table.select('td')[:10]))
        _today['minimum_temp'], _tomorrow['minimum_temp'], _after_tomorrow['minimum_temp'] = all_temps[2:5]
        _today['highst_temp'], _tomorrow['highst_temp'], _after_tomorrow['highst_temp'] = all_temps[7:10]

        return {'forecast':_forecast, 'today':_today, 'tomorrow':_tomorrow, 'after_tomorrow':_after_tomorrow}

class News:
    """
Functions:
:meth:`~openpibo.collect.News.search`

    JTBC 뉴스 RSS 서비스를 사용해 뉴스 자료를 가져옵니다.

    example::

        from openpibo.collect import News

        pibo_news = News()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
    """
    topic_list = {
        '속보': 'newsflash',
        '정치': 'politics',
        '경제': 'economy',
        '사회': 'society',
        '국제': 'international',
        '문화': 'culture',
        '연예': 'entertainment',
        '스포츠': 'sports',
        '풀영상': 'fullvideo',
        '뉴스랭킹': 'newsrank',
        '뉴스룸': 'newsroom',
        '아침&': 'morningand',
        '썰전 라이브': 'ssulzunlive',
        '정치부회의': 'politicaldesk',
    }
    """
    뉴스 정보를 검색할 수 있는 주제입니다.
    """

    def search(self, search_topic='뉴스랭킹'):
        """
        해당 주제(```search_topic```)에 맞는 뉴스를 가져옵니다.

        example::

            result = pibo_news.search('속보')

        :param str topic: 검색 가능한 뉴스 주제 (default: 뉴스랭킹)

            검색할 수 있는 주제는 다음과 같습니다::

                '속보', '정치', '경제', '사회', '국제', '문화', '연예', '스포츠',
                '풀영상', '뉴스랭킹', '뉴스룸', '아침&', '썰전 라이브', '정치부회의'

        :returns: title, link, description, pubDate 요소가 있는 dictionary 배열입니다.

            example::

                [
                    {
                        'title': '또 소방차 막은 불법주차, 이번엔 가차없이 밀어버렸다', 
                        'link': 'https://news.jtbc.joins.com/article/article.aspx?...',
                        'description': '2019년 4월 소방당국의 불법주정차 강경대응 훈련 모습...,
                        'pubDate': '2021.09.03'
                    }, 
                ]
                or None
        """
        topic = News.topic_list.get(search_topic)
        if topic == None:
          raise Exception(f'"{search_topic}" not support')

        _articles = []
        url = f'https://fs.jtbc.joins.com//RSS/{topic}.xml'
        res = requests.get(url, headers={'User-Agent' : 'Mozilla/5.0'})
        soup = BeautifulSoup(res.content, 'xml')
        items = soup.findAll('item')
        for item in items:
          _articles.append({
            'title':item.find('title').text,
            'link':item.find('link').text,
            'description':item.find('description').text,
            'pubDate':item.find('pubDate').text
          })
        return _articles


if __name__ == "__main__":
    # wiki = Wikipedia()
    # print(wiki.search("강아지"))
    # weather = Weather('제주')
    # print(weather)
    news = News()
    print(news.get_article(3))
