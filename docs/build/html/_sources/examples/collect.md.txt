# collect

## collect_test.py

각종 데이터를 수집해옵니다.

```python
from openpibo.collect import Wikipedia
from openpibo.collect import Weather
from openpibo.collect import News


def run():
  # 위키 검색
  wiki = Wikipedia()
  result = wiki.search('사과')
  print('=== Wikipedia ===')
  print('Result:', result['0'])

  # 날씨 검색
  weather = Weather()
  result = weather.search('서울')
  print('\n\n=== Weather ===')
  print('Keyword:', Weather.region_list.keys())
  print('Result:', result)

  # 뉴스 검색
  news = News()
  result = news.search('경제')
  print('\n\n=== News ===')
  print('Keyword:', News.topic_list.keys())
  print('Result:', result)

if __name__ == "__main__":
  run()
```

**collect_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/collect $ sudo python3 collect_test.py
```

**collect_test.py 결과**

다음 텍스트가 출력됩니다.

```
=== Wikipedia ===
Result: {'title': '개요', 'content': ['사과(沙果, apple)는 사과나무의 열매로, 세계적으로 널리 재배되는 열매 가운데 하나이다. 평과(苹果), 빈파(頻婆)라고도 한다.\n', '사과열매는 가을에 익는데, 보통 지름이 5~9센티미터이다. 극히 드물지만 15센티에 이르기도 한다. 씨앗에는 미량의 사이안화물이 함유되어 있다. 샐러드, 주스, 파이, 카레 등의 재료로 쓰인다.\n']}


=== Weather ===
Keyword: dict_keys(['전국', '서울', '인천', '경기', '부산', '울산', '경남', '대구', '경북', '광주', '전남', '전북', '대전', '세종', '충남', '충북', '강원', '제주'])
Result: {'forecast': '내일 새벽부터 낮 사이 가끔 비', 'today': {'weather': '대체로 흐림, 경기북부와 서해5도 오늘 밤까지 빗방울', 'minimum_temp': '16.2 ~ 20.4', 'highst_temp': '25.4 ~ 28.4'}, 'tomorrow': {'weather': '대체로 흐림, 새벽부터 낮 사이 가끔 비, 서해5도 새벽 한때 비', 'minimum_temp': '15 ~ 20', 'highst_temp': '17.9 ~ 24.9'}, 'after_tomorrow': {'weather': '대체로 흐림, 새벽부터 오전 사이 경기남부 가끔 비, 그 밖의 지역 빗방울', 'minimum_temp': '14 ~ 18', 'highst_temp': '21 ~ 23'}}


=== News ===
Keyword: dict_keys(['속보', '정치', '경제', '사회', '국제', '문화', '연예', '스포츠', '풀영상', '뉴스랭킹', '뉴스룸', '아침&', '썰전 라이브', '정치부회의'])
Result: [{'title': '떨어진 인삼값에 밭 갈아엎는 농민들…"생존권 보장을"', 'link': 'https://news.jtbc.joins.com/article/article.aspx?news_id=NB12025894', 'description': ' [앵커]인삼 가격이 너무 떨어져서 차라리 밭을 갈아엎는 게 낫다는 농민들이 있습니다. 정부가 대책을 만들어주길 요구하고 있습니다.정영재 기자입니다.[기자]산비탈 밭에 트랙터가 흙을 파내며 지나갑니다.인삼이', 'pubDate': '2021.10.06'},
....
```
