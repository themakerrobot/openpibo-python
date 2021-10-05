"""
인터넷에서 유용한 정보를 가져옵니다.

**단어정보, 날씨 정보, 뉴스 정보** 를 가져올 수 있습니다.

Class:
:obj:`~openpibo.collect.Wikipedia`
:obj:`~openpibo.collect.Weather`
:obj:`~openpibo.collect.News`
"""

from urllib.parse import quote
from .modules.collect.get_soup import get_soup


class _Chapter:
    """
    위키백과에서의 한 쳅터에 대한 클래스입니다.
    """

    def __init__(self, title):
        self._title = title
        self._content = ''
        self._parent = None

    def init_content(self):
        self._content = ''
    
    def add_content(self, content):
        self._content += content
    
    def set_parent(self, parent_node):
        self._parent = parent_node


class Wikipedia:
    """
:meth:`~openpibo.collect.Wikipedia.search`

    위키백과에서 단어를 검색합니다.

    example::

        from openpibo.collect import Wikipedia

        pibo_wiki = Wikipedia()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
    """

    def __init__(self):
        self._summary = ''
        self._chapters = {'0': _Chapter('개요')}
        self._titles = []
        
    def __str__(self):
        """클래스를 출력하면 첫 번째 쳅터(주로 '개요')의 내용을 출력합니다.
        
        example::

            print(pibo_wiki)
            >>> 강아지 (dog)는 개의 새끼를 일컫는다 ..."""

        return self._chapters['0']._content
    
    def search(self, search_text: str):
        """
        위키백과에서 ``search_text`` 를 검색합니다.
        
        print 인스턴스(pibo_wiki)로 개요 정보를 출력할 수 있습니다.
        
        example::

            pibo_wiki.search('강아지')
            print(pibo_wiki)
            # 강아지 (dog)는 개의 새끼를 일컫는다 ...
        
        만약 검색 결과가 없으면, 다음과 같이 출력됩니다::

            pibo_wiki.search('깡아지')
            print(pibo_wiki)
            # '깡아지'에 대한 검색결과가 없습니다.
        
        :param str search_text: 위키백과에서의 검색어
        
        :returns: 내용이 dictionary 배열 형태로 출력됩니다.

            example::

                [{
                    'title': '명칭', 
                    'content': "한국어 ‘강아지’는 ‘개’에 어린 짐승을 뜻하는 ‘아지’가 붙은 말이다..."
                }]
        """

        self._chapters = {'0': _Chapter('개요')}
        self._titles = []
        encode_text = quote(search_text)
        url = f'https://ko.wikipedia.org/wiki/{encode_text}'
        soup = get_soup(url)
        total_content = soup.find('div', {'class': 'mw-parser-output'})
        if not total_content:
            self._chapters['0'].add_content(f"'{search_text}'에 대한 검색결과가 없습니다.")
            return
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
                self._chapters[chapter_idx] = _Chapter(title=content.text.split('[')[0])
                self._titles.append(content.text)
            elif tag == 'p':
                self._chapters[chapter_idx].add_content(content.text)
            elif tag == 'ul':
                self._chapters[chapter_idx].add_content(content.text)
        
        return self._chapters

    def get_list(self):
        """
        챕터의 목록을 list 형태로 가져옵니다.
        
        example::

            pibo_wiki.get_list()
        
        :returns: list 형태의 챕터 목록입니다.

            list 안에 str 타입의 챕터 번호가 기록됩니다.

            example::
            
                ['0', '1', '2', '3', '4', '5', '6']
        """
        
        return list(self._chapters.keys())

    def get(self, chapter_num):
        """
        ``chapter_num`` 에 해당하는 내용을 출력합니다.
        
        example::
        
            pibo_wiki.get('1')

        :param str chapter_num: 챕터의 번호

            ``1.3.1`` 과 같이 표현되기 때문에 int 또는 float 타입이 아닌 **str 타입** 입니다.
        
        :returns: 해당 챕터 번호에 해당하는 내용이 dictionary 형태로 출력됩니다.

            example::

                {
                    'title': '명칭', 
                    'content': "한국어 ‘강아지’는 ‘개’에 어린 짐승을 뜻하는 ‘아지’가 붙은 말이다..."
                }
        """
        
        title = self._chapters[chapter_num]._title
        content = self._chapters[chapter_num]._content
        if not content:
            content = '(내용이 존재하지 않습니다.)'
        result = {}
        result['title'] = title
        result['content'] = content
        return result


region_table = {
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
    '제주': 184
}
class Weather:
    """
:meth:`~openpibo.collect.Weather.search`

    오늘, 내일, 모레의 날씨 정보를 가져옵니다.

    example::

        from openpibo.collect import Weather
    
        pibo_weather = Weather()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다."""

    def __init__(self):
        """해당 지역의 날씨를 가져옵니다."""

        self._region = None
        self._forecast = ""
        self._today = {}
        self._tomorrow = {}
        self._after_tomorrow = {}

    def __str__(self):
        """해당 지역의 단기예보를 반환합니다."""

        if self._forecast:
            return self._forecast
        return "검색된 지역이 없습니다. '.search()' 를 하십시오."
    
    def search(self, region:str='전국'):
        """
        해당 지역의 날씨 정보를 가져와서 인스턴스(pibo_weather)에 저장합니다.

        ``get_today``, ``get_tomorrow``, ``get_after_tomorrow`` 메소드로 날씨 정보를 출력할 수 있습니다.

        print 인스턴스(pibo_weather)로 전체적인 날씨를 출력할 수 있습니다.

        example::

            pibo_weather.search('서울')
            print(pibo_weather)
            # 내일 경기남부 가끔 비, 내일까지 바람 약간 강, 낮과 밤의 기온차 큼
        
        :param str region: 검색 하려는 지역 (default: 전국)

            검색할 수 있는 지역은 다음과 같습니다::
                
                '전국', '서울', '인천', '경기', '부산', '울산', '경남', '대구', '경북', 
                '광주', '전남', '전북', '대전', '세종', '충남', '충북', '강원', '제주'
        
        :returns: 오늘/내일/모레의 날씨 및 최저/최고기온을 반환합니다.
            
            example::

                {
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
        """

        self._region = region
        self._forecast = ''
        self._today = {}
        self._tomorrow = {}
        self._after_tomorrow = {}

        try:
            region_num = region_table[region]
        except:
            raise Exception(f"""'region'에 들어갈 수 있는 단어는 다음과 같습니다.\n\t{tuple(region_table.keys())}""")
        url = f'https://www.weather.go.kr/w/weather/forecast/short-term.do?stnId={region_num}'
        soup = get_soup(url)
        forecasts = soup.find('div', {'class': 'cmp-view-content'}).text
        forecasts = forecasts.split('□')[1].split('○')
        for forecast in map(str.strip, forecasts):
            split_point = forecast.index(')')
            date = forecast[:split_point+1]
            desc = forecast[split_point+2:]
            if '종합' in date:
                self._forecast = desc
            if '오늘' in date:
                self._today['weather'] = desc
            if '내일' in date or '~' in date:
                self._tomorrow['weather'] = desc
            if '모레' in date:
                self._after_tomorrow['weather'] = desc

        temp_table = soup.find('tbody')
        all_temps = list(map(lambda x: x.text, temp_table.select('td')[:10]))
        self._today['minimum_temp'], self._tomorrow['minimum_temp'], self._after_tomorrow['minimum_temp'] = all_temps[2:5]
        self._today['highst_temp'], self._tomorrow['highst_temp'], self._after_tomorrow['highst_temp'] = all_temps[7:10]
        
        return {'today':self._today, 'tomorrow':self._tomorrow, 'after_tomorrow':self._after_tomorrow}

    def get_today(self):
        """
        오늘의 날씨를 반환합니다.
        
        example::
        
            pibo_weather.get_today()
        
        :returns: 오늘의 날씨 및 최저/최고기온을 반환합니다.

            example::

                {
                    'weather': '전국 대체로 흐림',
                    'minimum_temp': '15.3 ~ 21.6', 
                    'highst_temp': '23.1 ~ 27.6'
                }
        """

        return self._today
    
    def get_tomorrow(self):
        """
        내일의 날씨를 반환합니다.
        
        example::
        
            pibo_weather.get_tomorrow()
        
        :returns: 내일의 날씨 및 최저/최고기온을 반환합니다.

            example::

                {
                    'weather': '전국 대체로 흐리고 비, 낮에 남부지방과 제주도부터 차차 그침', 
                    'minimum_temp': '16 ~ 24', 
                    'highst_temp': '20.8 ~ 26.6'
                }
        """

        return self._tomorrow
    
    def get_after_tomorrow(self):
        """
        모레의 날씨를 반환합니다.
        
        example::
        
            pibo_weather.get_after_tomorrow()
        
        :returns: 모레의 날씨 및 최저/최고기온을 반환합니다.

            example::

                {
                    'weather': '전국 대체로 흐리다가 오후에 서쪽지방부터 차차 맑아짐', 
                    'minimum_temp': '17 ~ 22', 
                    'highst_temp': '22 ~ 30'
                }
        """

        return self._after_tomorrow


topic_table = {
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
class News:
    """
:meth:`~openpibo.collect.News.search`

    JTBC 뉴스 RSS 서비스를 사용해 뉴스 자료를 가져옵니다.

    example::

        from openpibo.collect import News
    
        pibo_news = News()
        # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
    """

    def __init__(self):
        f"""'topic'에는 아래와 같은 단어가 들어갈 수 있습니다.\n\t{tuple(topic_table.keys())}"""

        self._topic = ''
        self._articles = []
        self._titles = []
    
    def __str__(self):
        """가장 최근 기사의 제목을 반환합니다."""

        if self._titles:
            return self._titles[0]
        return "검색된 주제가 없습니다. '.search()' 를 하십시오."
    
    def search(self, topic='뉴스랭킹'):
        """
        주제에 맞는 뉴스를 검색하여 인스턴스(pibo_news)에 저장합니다.

        print 인스턴스(pibo_news)로 첫 번째 뉴스 헤드라인을 출력할 수 있습니다.

        example::

            pibo_news.search('속보')
            print(pibo_news)
            # JTBC, 파일럿 예능-특선 영화로 꽉 채운 추석 라인업 공개
        
        :param str topic: 검색할 뉴스 주제

            다음과 같은 단어가 들어갈 수 있습니다.

            * ``속보``
            * ``정치``
            * ``경제``
            * ``사회``
            * ``국제``
            * ``문화``
            * ``연예``
            * ``스포츠``
            * ``풀영상``
            * ``뉴스랭킹``
            * ``뉴스룸``
            * ``아침&``
            * ``썰전 라이브``
            * ``정치부회의``

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
        """

        self._topic = topic
        self._articles = []
        self._titles = []

        topic_code = topic_table[topic]
        url = f'https://fs.jtbc.joins.com//RSS/{topic_code}.xml'
        soup = get_soup(url, 'xml')
        items = soup.findAll('item')
        for item in items:
            title = item.find('title').text
            link = item.find('link').text
            description = item.find('description').text
            pubDate = item.find('pubDate').text
            article = {
                'title': title,
                'link': link,
                'description': description,
                'pubDate': pubDate
                }
            self._articles.append(article)
            self._titles.append(title)
        return self._articles  
    
    def get_titles(self):
        """
        기사 제목 목록을 모두 보여줍니다.

        example::

            pibo_news.get_titles()
        
        :returns: ``key=기사번호`` , ``value=title`` 인 dictionary 입니다.

            example::

                {
                    0: '또 소방차 막은 불법주차, 이번엔 가차없이 밀어버렸다', 
                    1: '"죽은 줄 알았던 11살 아들, 40살이 돼 돌아왔습니다"', 
                    2: "이런 영웅은 처음이지?…마블 '아시아 히어로' 통할까", 
                    ...
                    19: "[크로스체크] 5시간 새 100배 증식…'식중독 주범' 살모넬라균 추적"
                }
        """

        article_mapping = {}
        for idx, title in enumerate(self._titles):
            article_mapping[idx] = title
        return article_mapping
    
    def get_article(self, article_idx):
        """
        기사 번호에 해당하는 기사 정보를 보여줍니다.

        example::

            pibo_news.get_article(1)
        
        :param int article_idx: 기사 번호

            기사 번호는 0~19 사이 int 타입 입니다.
        
        :returns: title, link, description, pubDate 요소가 있는 dictionary 입니다.

            example::

                {
                    'title': '또 소방차 막은 불법주차, 이번엔 가차없이 밀어버렸다', 
                    'link': 'https://news.jtbc.joins.com/article/article.aspx?...',
                    'description': '2019년 4월 소방당국의 불법주정차 강경대응 훈련 모습...,
                    'pubDate': '2021.09.03'
                }
        """

        return self._articles[article_idx]


if __name__ == "__main__":
    # wiki = Namuwiki("강아지")
    # print(wiki)
    # weather = Weather('제주')
    # print(weather)
    news = News()
    print(news.get_article(3))
