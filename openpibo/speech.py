"""
번역, 형태소 분석, 자연어 인식 및 합성, 챗봇 등 다양한 자연어 처리를 합니다.

Class:
:obj:`~openpibo.speech.Speech`
:obj:`~openpibo.speech.Dialog`
"""

import csv
import random
#import io
import json
import os
from konlpy.tag import Mecab
import requests
from . import napi_host, sapi_host
from .modules.speech.mtranslate import translate
import openpibo_models
#current_path = os.path.dirname(os.path.realpath(__file__))

class Speech:
  """
Functions:
:meth:`~openpibo.speech.Speech.tts`
:meth:`~openpibo.speech.Speech.stt`

  * TTS (Text to Speech)
  * STT (Speech to Text)

  example::

    from openpibo.speech import Speech

    pibo_speech = Speech()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.SAPI_HOST = sapi_host

  def tts(self, string, filename="tts.mp3", voice="main", lang="ko"):
    """
    TTS(Text to Speech)

    Text(문자)를 Speech(말)로 변환하여 파일로 저장합니다.

    example::

      pibo_speech.tts('안녕하세요! 만나서 반가워요!', 'main', 'ko', '/home/pi/tts.mp3')

    :param str string: 변환할 문장구

    :param str voice: 목소리 타입(main | boy | girl | man1 | woman1)

    :param str lang: 사용할 언어(ko | en)

    :param str filename: 변환된 음성파일의 경로 (mp3)
    """

    if type(string) is not str:
      raise Exception(f'"{string}" must be str type')

    if type(voice) is not str or voice not in ('espeak', 'main', 'boy', 'girl', 'man1', 'woman1'):
      raise Exception(f'"{voice}" must be (espeak|main|boy|girl|man1|woman1)')

    if type(lang) is not str or lang not in ('en', 'ko'):
      raise Exception(f'"{lang}" must be (en|ko)')

    if voice == "espeak":
      os.system(f'esspeak {string} -w {filename}')
      return

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


class Dialog:
  """
Functions:
:meth:`~openpibo.speech.Dialog.load`
:meth:`~openpibo.speech.Dialog.mecab_pos`
:meth:`~openpibo.speech.Dialog.mecab_morphs`
:meth:`~openpibo.speech.Dialog.mecab_nouns`
:meth:`~openpibo.speech.Dialog.ngram`
:meth:`~openpibo.speech.Dialog.diff_ngram`
:meth:`~openpibo.speech.Dialog.get_dialog`
:meth:`~openpibo.speech.Dialog.translate`
:meth:`~openpibo.speech.Dialog.get_dialog_dl`
:meth:`~openpibo.speech.Dialog.nlp_dl`

  파이보에서 대화와 관련된 자연어처리 기능을 하는 클래스입니다. 다음 기능을 수행할 수 있습니다.

  * 형태소 및 명사 분석
  * 챗봇 기능
  * 한역 번역 / 대화 (Deep Learning) 
  * 자연어 분석 기능 (Deep Learning)

  example::

    from openpibo.speech import Dialog

    pibo_dialog = Dialog()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.dialog_db = []
    self.mecab = Mecab()
    self.NAPI_HOST = napi_host
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
      self.dialog_db = [item for item in csv.reader(f)]

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

  def ngram(self, string, n=2):

    """
    N-gram 값을 구합니다.

    exmaple::

      pibo_dialog.ngram('아버지가 방에 들어가셨다.')
      # ['아버', '버지', '지가', '가 ', ' 방', '방에', '에 ', ' 들', '들어', '어가', '가셨', '셨다', '다.']

    :param str string: 분석할 문장 (한글)
    
    :param int n: N-gram에 사용할 n 값 (한글) default:2

    :returns: 문장에서 추출한 N-gram 값

      ``list`` 타입 입니다.
    """

    return [string[i:i+n] for i in range(len(string)-n+1)]

  def diff_ngram(self, string_a, string_b, n=2):

    """
    N-gram 방식으로 두 문장을 비교하여 유사도를 구합니다.

    exmaple::

      pibo_dialog.diff_ngram('아버지가 방에 들어가셨다.' '어머니가 방에 들어가셨다.')
      # 0.6923076923076923

    :param str string: 비교할 문장A (한글)
    
    :param str string: 비교할 문장B (한글)
    
    :param int n: N-gram에 사용할 n 값 (한글) default:2

    :returns: N-gram 방식으로 비교한 유사도

      ``float`` 타입 입니다.
    """

    a = self.ngram(string_a, n)
    b = self.ngram(string_b, n)

    cnt = 0
    for i in a:
      for j in b:
        if i == j:
          cnt += 1
    return cnt / len(a)

  def get_dialog(self, q, n=2):
    """
    일상대화에 대한 답을 추출합니다.

    저장된 데이터로부터 사용자의 질문과 가장 유사한 질문을 선택해 그에 대한 답을 출력합니다.

    example::

      pibo_dialog.get_dialog('나랑 같이 놀자')

    :param str string: 질문하는 문장 (한글)
    
    :param int n: N-gram에 사용할 n 값 (한글) default:2

    :returns: 답변하는 문장 (한글)

      ``string`` 타입 입니다.
    """

    max_acc = 0
    max_ans = []
    for line in self.dialog_db:
      acc = self.diff_ngram(q, line[0], n)

      if acc == max_acc:
        max_ans.append(line)

      if acc > max_acc:
        max_acc = acc
        max_ans = [line]

    return random.choice(max_ans)[1]

  def translate(self, string, target="en"):
    """
    문장을 번역합니다.

    example::

      pibo_dialog.translate('안녕하세요! 만나서 정말 반가워요!')
      # "Hi! Nice to meet you!"

    :param str string: 번역할 문장

    :param str target: 번역될 언어(ko, en, ja, fr ...)

    :returns: 번역 된 문장
    """

    if type(string) is not str or type(target) is not str:
      raise Exception(f'"{string}, {target}" must be str type')

    return translate(string, target)

  def get_dialog_dl(self, string):
    """
    일상대화에 대한 답을 추출합니다.(Deep Learning)

    example::

      pibo_dialog.get_dialog_ml('나랑 같이 놀자')

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

  def nlp_dl(self, string, mode):
    """
    문장을 분석합니다.(Deep Learning)

    문장을 지정한 모드로 분석합니다.

    example::

      pibo_dialog.nlp_ml('안녕하세요. 오늘 매우 즐거워요', 'ner')

    :param str string: 분석할 문장

    :param str mode: 분석 모드 `` (summary|vector|sentiment|emotion|ner|wellness|hate) ``

    :returns: 분석 결과
    """
    #if type(mode) is not str or mode not in ('summary', 'vector', 'sentiment', 'emotion', 'ner', 'wellness', 'hate'):
    #  raise Exception(f'"{mode}" must be (summary|vector|sentiment|emotion|ner|wellness|hate)')

    res = requests.post(self.NAPI_HOST + '/' + mode, params={"sentence":string})

    if res.status_code != 200:
      raise Exception(f'response error: {res}')

    if res.json()['result'] == False:
      raise Exception(f'result error: {res.json()}')

    return res.json()['data']
