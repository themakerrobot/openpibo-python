"""
번역, 형태소 분석, 자연어 인식 및 합성, 챗봇 등 다양한 자연어 처리를 합니다.

Class:
:obj:`~openpibo.speech.Speech`
:obj:`~openpibo.speech.Dialog`
:obj:`~openpibo.speech.Speech2`
:obj:`~openpibo.speech.Dialog2`
"""

import csv
import random
#import io
import json
import os
from konlpy.tag import Mecab
import requests
from .modules.speech.google_trans_new import google_translator
from . import kakaokey, napi_host, sapi_host

import openpibo_models
#current_path = os.path.dirname(os.path.realpath(__file__))


class Speech:
  """
Functions:
:meth:`~openpibo.speech.Speech.translate`
:meth:`~openpibo.speech.Speech.tts`
:meth:`~openpibo.speech.Speech.stt`

  Kakao 음성 API를 사용하여 사람의 음성 언어를 인식, 합성하거나 Google 번역 모듈을 사용하여 번역을 합니다.

  * 번역 (한국어, 영어)
  * TTS (Text to Speech)
  * STT (Speech to Text)

  ``config.json`` 의 ``kakaokey`` 에 본인의 ``KAKAO REST API KEY`` 를 입력해야 사용할 수 있습니다.

  example::

    from openpibo.speech import Speech

    pibo_speech = Speech()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.translator = google_translator()
    self.kakao_account = kakaokey

  def translate(self, string, to='ko'):
    """
    구글 번역기를 이용해서 문장을 번역합니다.

    example::

      pibo_speech.translate('안녕하세요! 만나서 정말 반가워요!', to='en')
      # "Hello! I'm really happy to meet you!"

    :param str string: 번역할 문장

    :param str to: 번역될 언어

      ``en`` 또는 ``ko``

    :returns: 번역 된 문장
    """

    '''curl -v -X POST "https://dapi.kakao.com/v2/translation/translate" \
    -d "src_lang=kr" \
    -d "target_lang=en" \
    --data-urlencode "query=안녕하세요, 반갑습니다." \
    -H "Authorization: KakaoAK {REST_API_KEY}"'''

    '''
    # kakao translate source
    url = 'https://dapi.kakao.com/v2/translation/translate'
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Authorization': 'KakaoAK ' + self.kakao_account
    }

    res = requests.post(url, headers=headers, data={"src_lang":"kr", "target_lang":"en", "query":string})
    try:
      result = {"result":True, "value":json.loads(res.text)["translated_text"]}

    except Exception as ex:
      result = {"result":False, "value":""}
    return result['value']'''

    if type(string) is not str:
      raise Exception(f'"{string}" must be str type')
    
    if type(to) is not str or to not in ('en', 'ko'):
      raise Exception(f'"{to}" must be (en|ko)')

    return self.translator.translate(string, lang_tgt=to)

  def tts(self, string, filename="tts.mp3"):
    """
    TTS(Text to Speech)

    Text(문자)를 Speech(말)로 변환하여 파일로 저장합니다.

    example::

      pibo_speech.tts('안녕하세요! 만나서 반가워요!', '/home/pi/tts.mp3')

    :param str string: 변환할 문구

    :param str filename: 변환된 음성파일의 경로

      파일의 확장자에는 제한이 없지만, 파이보에서 재생하기 위해서 ``mp3`` 또는 ``wav`` 확장자를 추천합니다.
    """

    '''curl -v "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize" \
    -H "Content-Type: application/xml" \
    -H "Authorization: KakaoAK API_KEY" \
    -d '<speak> 그는 그렇게 말했습니다.
    <voice name="MAN_DIALOG_BRIGHT">잘 지냈어? 나도 잘 지냈어.</voice>
    <voice name="WOMAN_DIALOG_BRIGHT" speechStyle="SS_ALT_FAST_1">금요일이 좋아요.</voice> </speak>' > result.mp3'''

    if self.kakao_account in [None, '']:
      raise Exception('Kakao account invalid')

    url = "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize"
    headers = {
      'Content-Type': 'application/xml',
      'Authorization': 'KakaoAK ' + self.kakao_account
    }
    r = requests.post(url, headers=headers, data=string.encode('utf-8'))
    with open(filename, 'wb') as f:
      f.write(r.content)

  def stt(self, filename="stream.wav", timeout=5):
    """
    STT(Speech to Text)

    목소리를 녹음한 후 파일로 저장하고, 그 파일의 Speech(말)를 Text(문자)로 변환합니다.

    녹음 파일은 ``timeout`` 초 동안 녹음되며, ``filename`` 의 경로에 저장됩니다.

    example::

      pibo_speech.stt('/home/pi/stt.wav', 5)

    :param str filename: 녹음한 파일이 저장 될 경로. ``wav`` 확장자를 사용합니다.

    :param int timeout: 녹음 시간(s)

    :returns: ``True`` / ``False``
    """

    if self.kakao_account in [None, '']:
      raise Exception('Kakao account invalid')

    os.system(f'arecord -D dmic_sv -c2 -r 16000 -f S32_LE -d {timeout} -t wav -q -vv -V streo stream.raw;sox stream.raw -c 1 -b 16 {filename};rm stream.raw')

    '''curl -v "https://kakaoi-newtone-openapi.kakao.com/v1/recognize" \
    -H "Transfer-Encoding: chunked" -H "Content-Type: application/octet-stream" \
    -H "Authorization: KakaoAK API_KEY" \
    --data-binary @stream.wav '''
    url = 'https://kakaoi-newtone-openapi.kakao.com/v1/recognize'
    headers = {
      'Content-Type': 'application/octet-stream',
      'Authorization': 'KakaoAK ' + self.kakao_account
    }

    with open(filename, 'rb') as f:
      data = f.read()
    res = requests.post(url, headers=headers, data=data)
    try:
      result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}')+1]
    except Exception as ex:
      result_json_string = res.text[res.text.index('{"type":"errorCalled"'):res.text.rindex('}')+1]
    result = json.loads(result_json_string)
    return result['value']


class Dialog:
  """
Functions:
:meth:`~openpibo.speech.Dialog.load`
:meth:`~openpibo.speech.Dialog.mecab_pos`
:meth:`~openpibo.speech.Dialog.mecab_morphs`
:meth:`~openpibo.speech.Dialog.mecab_nouns`
:meth:`~openpibo.speech.Dialog.conversation`
:meth:`~openpibo.speech.Dialog.get_distance`

  파이보에서 대화와 관련된 자연어처리 기능을 하는 클래스입니다. 다음 기능을 수행할 수 있습니다.

  * 형태소 및 명사 분석
  * 챗봇 기능

  example::

    from openpibo.speech import Dialog

    pibo_dialog = Dialog()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.dialog_db = []
    self.mecab = Mecab()
    self.load(openpibo_models.filepath("dialog.csv"))

  def load(self, filepath):
    """
    대화 데이터를 로드합니다.

    example::

      pibo_dialog.load('/home/pi/dialog.csv')

    :param str string: 대화 데이터 파일 경로(csv)

    대화 데이터 파일 형식::

      대화1,답변1
      대화2,답변2
      ...
      대화n,답변n
    """

    self.dialog_path = filepath
    with open(self.dialog_path, 'r', encoding='utf-8') as f:
      rdr = csv.reader(f)
      self.dialog_db = [[self.mecab_morphs(line[0]), line[1]] for line in rdr]

  def mecab_pos(self, string):
    """
    형태소를 품사와 함께 추출합니다.

    exmaple::

      pibo_dialog.mecab_pos('아버지가 방에 들어가셨다.')
      # [('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('셨', 'EP+EP'), ('다', 'EF'), ('.', 'SF')]

    :param str string: 분석할 문장 (한글)

    :returns: 형태소 분석 결과

      ``list(형태소, 품사)`` 형태로 출력됩니다.
    """

    return self.mecab.pos(string)

  def mecab_morphs(self, string):

    """
    형태소를 추출합니다.

    exmaple::

      pibo_dialog.mecab_morphs('아버지가 방에 들어가셨다.')
      # ['아버지', '가', '방', '에', '들어가', '셨', '다', '.']

    :param str string: 분석할 문장 (한글)

    :returns: 형태소 분석 결과

      ``list`` 타입 입니다.
    """

    return self.mecab.morphs(string)

  def mecab_nouns(self, string):

    """
    명사를 추출합니다.

    exmaple::

      pibo_dialog.mecab_nouns('아버지가 방에 들어가셨다.')
      # ['아버지', '방']

    :param str string: 분석할 문장 (한글)

    :returns: 문장에서 추출한 명사 목록

      ``list`` 타입 입니다.
    """

    return self.mecab.nouns(string)

  def get_dialog(self, q):
    """
    일상대화에 대한 답을 추출합니다.

    저장된 데이터로부터 사용자의 질문과 가장 유사한 질문을 선택해 그에 대한 답을 출력합니다.

    example::

      pibo_dialog.get_dialog('나랑 같이 놀자')

    :param str string: 질문하는 문장 (한글)

    :returns: 답변하는 문장 (한글)

      ``string`` 타입 입니다.
    """
    def get_distance(aT, bT):
      cnt = 0
      for i in aT:
        for j in bT:
          if i == j:
            cnt += 1
      return cnt / len(aT)

    max_acc = 0
    max_ans = []
    c = self.mecab_morphs(q)
    for line in self.dialog_db:
      acc = get_distance(line[0], c)

      if acc == max_acc:
        max_ans.append(line)

      if acc > max_acc:
        max_acc = acc
        max_ans = [line]

    return random.choice(max_ans)[1]


class Speech2:
  """
Functions:
:meth:`~openpibo.speech.Speech2.translate`
:meth:`~openpibo.speech.Speech2.tts`
:meth:`~openpibo.speech.Speech2.stt`

  * 번역 (한국어 -> 영어)
  * TTS (Text to Speech)
  * STT (Speech to Text)

  example::

    from openpibo.speech import Speech2

    pibo_speech2 = Speech2()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.SAPI_HOST = sapi_host
    self.NAPI_HOST = napi_host

  def translate(self, string):
    """
    한글을 영어로 번역합니다.

    example::

      pibo_speech2.translate('안녕하세요! 만나서 정말 반가워요!')
      # "Hi! Nice to meet you!"

    :param str string: 번역할 문장 (한글)

    :returns: 번역 된 문장
    """

    if type(string) is not str:
      raise Exception(f'"{string}" must be str type')

    res = requests.post(self.NAPI_HOST + '/translation', params={"sentence":string})
    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    if res.json()['result'] == False:
      raise Exception(f'result error: {res.json()}')

    return res.json()['data']


  def tts(self, string, voice="main", lang="ko", filename="tts.mp3"):
    """
    TTS(Text to Speech)

    Text(문자)를 Speech(말)로 변환하여 파일로 저장합니다.

    example::

      pibo_speech2.tts('안녕하세요! 만나서 반가워요!', 'main', 'ko', '/home/pi/tts.mp3')

    :param str string: 변환할 문장구

    :param str voice: 목소리 타입(main | boy | girl | man1 | woman1)

    :param str lang: 사용할 언어(ko | en)

    :param str filename: 변환된 음성파일의 경로 (mp3)
    """

    if type(string) is not str:
      raise Exception(f'"{string}" must be str type')

    if type(voice) is not str or voice not in ('main', 'boy', 'girl', 'man1', 'woman1'):
      raise Exception(f'"{voice}" must be (main|boy|girl|man1|woman1)')

    if type(lang) is not str or lang not in ('en', 'ko'):
      raise Exception(f'"{lang}" must be (en|ko)')

    headers = {
      'accept': '*/*',
      'Content-Type': 'application/json',
    }

    data = {
      "text":string,
      "hash":"",
      "voice":voice, # ['main', 'boy', 'girl', 'man1', 'woman1']
      "lang":lang, # ['ko', 'en']
      "type":"mp3"
    }

    res = requests.post(self.SAPI_HOST + '/tts', headers=headers, json=data)
    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    with open(filename, 'wb') as f:
      f.write(res.content)


  def stt(self, filename="stream.wav", timeout=5):
    """
    STT(Speech to Text)

    목소리를 녹음한 후 파일로 저장하고, 그 파일의 Speech(말)를 Text(문자)로 변환합니다.

    녹음 파일은 ``timeout`` 초 동안 녹음되며, ``filename`` 의 경로에 저장됩니다.

    example::

      pibo_speech2.stt('/home/pi/stt.wav', 5)

    :param str filename: 녹음한 파일이 저장 될 경로. ``wav`` 확장자를 사용합니다.

    :param int timeout: 녹음 시간(s)

    :returns: 인식된 문자열
    """

    os.system(f'arecord -D dmic_sv -c2 -r 16000 -f S32_LE -d {timeout} -t wav -q -vv -V streo stream.raw;sox stream.raw -c 1 -b 16 {filename};rm stream.raw')

    res = requests.post(self.SAPI_HOST + '/stt', files={'file':open(filename, 'rb')})

    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    if res.json()['result'] == False:
      raise Exception(f'result error: {res.json()}')

    return res.json()['data']


class Dialog2:
  """
Functions:
:meth:`~openpibo.speech.Dialog2.load`
:meth:`~openpibo.speech.Dialog2.get_dialog`
:meth:`~openpibo.speech.Dialog2.nlp`

  파이보에서 대화와 관련된 자연어처리 기능을 하는 클래스입니다. 다음 기능을 수행할 수 있습니다.

  * 챗봇 기능
  * 자연어 분석 기능

  example::

    from openpibo.speech import Dialog2

    pibo_dialog = Dialog2()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.NAPI_HOST = napi_host

  def get_dialog(self, string):
    """
    일상대화에 대한 답을 추출합니다.

    example::

      pibo_dialog2.get_dialog('나랑 같이 놀자')

    :param str string: 질문하는 문장 (한글)

    :returns: 답변하는 문장 (한글)
    """
    res = requests.get(self.NAPI_HOST + '/dialog', params={'input':string})

    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    if res.json()['result'] == False:
      raise Exception(f'result error: {res.json()}')

    ans, score = [], []
    for item in res.json()['data']:
      ans.append(item['answer'])
      score.append(item['score'])

    return ans[score.index(max(score))]

  def nlp(self, string, mode):
    """
    문장을 분석합니다.

    문장을 지정한 모드로 분석합니다.

    example::

      pibo_dialog2.nlp('안녕하세요. 오늘 매우 즐거워요', 'ner')

    :param str string: 분석할 문장

    :param str mode: 분석 모드 `` (summary|vector|sentiment|emotion|ner|wellness|hate) ``

    :returns: 분석 결과
    """
    if type(mode) is not str or mode not in ('summary', 'vector', 'sentiment', 'emotion', 'ner', 'wellness', 'hate'):
      raise Exception(f'"{mode}" must be (summary|vector|sentiment|emotion|ner|wellness|hate)')

    res = requests.post(self.NAPI_HOST + '/' + mode, params={"sentence":string})

    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    if res.json()['result'] == False:
      raise Exception(f'result error: {res.json()}')

    return res.json()['data']

