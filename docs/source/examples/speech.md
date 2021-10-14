# speech

## chatbot_test.py

입력한 문장에 대해 형태소 분석을 실시하여 파이보와 대화를 시작합니다.

사용자가 입력한 문장에 db의 key 값(날씨, 음악, 뉴스)이 있으면 해당 함수를 실행하고, 없다면 대화봇을 실행합니다.

```python
from openpibo.speech import Dialog

def weather(cmd):
  topic = None

  # 분석한 문장 중 "오늘", "내일"이 있다면 topic 으로 설정
  for item in ['오늘', '내일']:
    if item in cmd:
      topic = item

  answer = f'{topic} 날씨 알려줄게요' if topic else '오늘, 내일 날씨만 가능해요.'
  print(f'날씨 > {answer}')

def music(cmd):
  topic = None

  # 분석한 문장 중 "발라드", "댄스", "락"이 있다면 topic 으로 설정
  for item in ['발라드', '댄스', '락']:
    if item in cmd:
      topic = item

  answer = f'{topic} 음악 들려줄게요' if topic else '발라드, 댄스, 락 음악만 가능해요.'
  print(f'음악 > {answer}')

def news(cmd):
  topic = None

  # 분석한 문장 중 "경제", "스포츠", "문화"가 있다면 topic 으로 설정
  for item in ['경제', '스포츠', '문화']:
    if item in cmd:
      topic = item

  answer = f'{topic} 뉴스 들려줄게요' if topic else '경제, 스포츠, 문화 뉴스만 가능해요.'
  print(f'뉴스 > {answer}')

func = {
  "날씨":weather,
  "음악":music, 
  "뉴스":news,
}

# 사용자가 입력한 문장에 대해 형태소 분석을 실시하여 파이보가 실행하는 함수가 달라짐
def run():
  o = Dialog()
  print("대화 시작합니다.")
  while True:
    c = input("입력 > ")
    matched = False
    if c == "그만":
      break

    # 사용자가 입력한 질문에 대한 형태소 분석
    d = o.mecab_morphs(c)
    # print("형태소 분석: ", d)
    # 분석한 문장 중 "날씨", "음악", "뉴스"가 있다면 해당 key값의 함수 실행
    for key in func.keys():
      if key in d:
        func[key](d)
        matched = True

    # key 값이 없다면 대화봇 실행
    if matched == False:
      print(f'대화 > {o.get_dialog(c)}')

if __name__ == "__main__":
  run()
```

![](images/speech_chatbot_flow.png)

**chatbot_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/speech $ sudo python3 chatbot_test.py 
```

**chatbot_test.py 결과**

```shell
대화 시작합니다.
입력 > 댄스 음악 추천해줘
음악 > 댄스 음악 들려줄게요.
입력 > 주말에 뭐하지
대화 > 사탕 만들어요.
입력 > 사탕 싫어
대화 > 싫어하지 말아요.
입력 > 그만
```

## mecab_test.py

사용자가 입력한 문장을 분석합니다. 3가지 모드 선택이 가능합니다.

```python
from openpibo.speech import Dialog

# mode(pos, morphs, nouns)에 따른 문장 분석
def run():
  o = Dialog()
  string = "아버지 가방에 들어가신다"
  
  print("입력: ", string)
  
  result = o.mecab_pos(string)
  print(f' pos: {result}')
  result = o.mecab_morphs(string)
  print(f' morphs: {result}')
  result = o.mecab_nouns(string)
  print(f' nouns: {result}')

if __name__ == "__main__":
  run()
```

**mecab_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/speech $ sudo python3 mecab_test.py 
```

**mecab_test.py 결과**

```shell
입력:  아버지 가방에 들어가신다
 pos: [('아버지', 'NNG'), ('가방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('신다', 'EP+EC')]
 morphs: ['아버지', '가방', '에', '들어가', '신다']
 nouns: ['아버지', '가방']
```

- NNG: 일반 명사 / JKB: 부사격 조사 / VV: 동사 / EP: 선어말 어미 / EC: 연결 어미

  ( 품사 태그표: [https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0](https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0))

## stt_test.py

```python
from openpibo.speech import Speech

def run():
  o = Speech()
  # 음성 언어를 문자 데이터로 변환하여 출력
  result = o.stt()
  print(result)

if __name__ == "__main__":
  run()
```

**stt_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/speech $ sudo python3 stt_test.py 
```

**stt_test.py 결과**

5초간 녹음된 음성을 텍스트로 변환하여 출력됩니다.

## translate_test.py

문장을 번역합니다.

```python
from openpibo.speech import Speech

# "즐거운 하루 보내세요."를 영어로 번역 후 출력
def run():
  o = Speech()
  string = "즐거운 하루 보내세요."
 
  print(f'입력 > {string}')

  result = o.translate(string, to="en")
  print(f' ko->en > {result}')
  result = o.translate(string, to="ko")
  print(f' en->ko > {result}')

if __name__ == "__main__":
  run()
```

**translate_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/speech $ sudo python3 translate_test.py 
```

**translate_test.py 결과**

```shell
입력 > 즐거운 하루 보내세요.
 ko->en > Have a nice day. 
 en->ko > 즐거운 하루 보내세요. 
```

## tts_test.py

문자 데이터를 음성 언어로 변환합니다.

```python
import openpibo
from openpibo.speech import Speech
from openpibo.audio import Audio

# tts.mp3 파일의 문자 데이터를 음성 언어로 변환 후, 파이보 스피커에 출력
def run():
  o_speech = Speech()
  o_audio = Audio()

  filename = openpibo.config['DATA_PATH']+"/audio/tts.mp3"
  o_speech.tts("<speak>\
              <voice name='MAN_READ_CALM'>안녕하세요. 반갑습니다.<break time='500ms'/></voice>\
            </speak>"\
          , filename)
  o_audio.play(filename, out='local', volume=-1500)  # 파이보 스피커로 filename 출력

if __name__ == "__main__":
  run()
```

- speak

  - 기본적으로 모든 음성은 태그로 감싸져야 한다.
  - 태그 하위로 `,`를 제외한 모든 태그가 존재할 수 있다.
  - 문장, 문단 단위로 적용하는 것을 원칙으로 한다. 한 문장 안에서 단어별로 태그를 감싸지 않는다.

  ```
  <speak> 안녕하세요. 반가워요. </speak>
  ```

- voice

  - 음성의 목소리를 변경하기 위해 사용하며, name attribute를 통해 원하는 목소리를 지정한다. 제공되는 목소리는 4가지이다.
    - WOMAN_READ_CALM: 여성 차분한 낭독체 (default)
    - MAN_READ_CALM: 남성 차분한 낭독체
    - WOMAN_DIALOG_BRIGHT: 여성 밝은 대화체
    - MAN_DIALOG_BRIGHT: 남성 밝은 대화체
  - 하위로 `,`를 제외한 모든 태그(kakao: effet, prosody, break, audio, say-as, sub)가 존재할 수 있다.
  - 문장, 문단 단위로 적용하는 것을 원칙으로 한다. 한 문장 안에서 단어별로 태그를 감싸지 않는다.

  ```
  <speak>
  <voice name="WOMAN_READ_CALM"> 지금은 여성 차분한 낭독체입니다.</voice>
  <voice name="MAN_READ_CALM"> 지금은 남성 차분한 낭독체입니다.</voice>
  <voice name="WOMAN_DIALOG_BRIGHT"> 안녕하세요. 여성 밝은 대화체예요.</voice>
  <voice name="MAN_DIALOG_BRIGHT"> 안녕하세요. 남성 밝은 대화체예요.</voice>
  </speak>
  ```

**tts_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/speech $ sudo python3 tts_test.py 
```

**tts_test.py 결과**

남성의 목소리로 `안녕하세요, 반갑습니다.` 라고 출력됩니다.
