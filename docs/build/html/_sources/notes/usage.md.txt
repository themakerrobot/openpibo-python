# 사용법

openpibo 패키지를 사용하여 파이보를 제어하는 방법과 데이터 경로를 입력하는 방법에 관해 설명합니다.

파이보의 기능을 사용하는 방법은 **클래스 호출 -> 인스턴스 생성 -> 메소드 사용** 순으로 이루어집니다.

## 클래스 호출 및 인스턴스 생성

클래스란, 정해진 기능을 사용할 수 있는 인스턴스를 생성하는 도구입니다.

클래스를 호출하여 인스턴스를 생성하는 방법은 다음과 같습니다.

```python
from openpibo.<라이브러리 명> import <클래스 명>

<인스턴스 명> = <클래스 명>()
```

예) audio 라이브러리의 Audio 클래스를 호출하여 인스턴스를 생성하는 방법은 다음과 같습니다.

```python
from openpibo.audio import Audio

pibo_audio = Audio()
```

인스턴스 명은 사용자 임의로 정할 수 있습니다.

## 메소드 사용

메소드란, 클래스에 묶여서 인스턴스와 관계되는 일을 하는 기능을 의미합니다.
예를 들어, Audio 클래스는 음악 재생, 정지 등의 메소드를 사용할 수 있습니다.

메소드를 사용하는 방법은 다음과 같습니다.

```python
<인스턴스 명>.<메소드명>(<인자>)
```

예) audio 라이브러리의 Audio 클래스 메소드를 사용하는 방법은 다음과 같습니다.

```python
# play 메소드: 오디오 파일을 재생합니다.
pibo_audio.play('/home/pi/openpibo_files/audio/test.mp3')

# stop 메소드: 재생 중인 오디오 파일을 중지합니다.
pibo_audio.stop()

# mute 메소드: 음소거 모드로 전환합니다.
pibo_audio.mute(True)

# pibo_audio 는 Audio 클래스의 인스턴스
```

클래스마다 사용할 수 있는 메소드는 각기 다르며, 메소드마다 입력되는 인자 또한 각기 다릅니다.

## 데이터 경로 설정

어떤 메소드는 인자로 파일 경로를 입력해야 합니다.

파일 경로를 입력하는 방법에 관해 설명하며,  
추가로 /home/pi/openpibo_files/audio/ 경로에 있는 test.mp3 파일 재생 방법을 예시로 활용합니다.

1. 절대 경로를 사용하는 방법

   절대 경로란, 파일이 가지고 있는 고유한 경로를 말합니다. 경로가 최상위 디렉터리부터 시작되는 특징이 있습니다.

   test.mp3 파일을 재생하는 메소드는 다음과 같습니다.

   ```python
   pibo_audio.play('/home/pi/openpibo_files/audio/test.mp3')
   # pibo_audio 는 Audio 클래스의 인스턴스
   ```

   위 예시에서 `test.mp3`의 경로는 `'/home/pi/openpibo_files/audio/test.mp3'` 입니다.

2. 상대 경로를 사용하는 방법

   상대 경로란, 현재 위치한 디렉터리를 기준으로 해서 대상 파일의 상대적인 경로를 의미합니다.

   현재 디렉터리 위치가 /home/pi/ 일때 test.mp3 파일의 상대 경로는 다음과 같습니다.

   ```python
   # 현재 디렉터리: /home/pi/
   pibo_audio.play('openpibo_files/audio/test.mp3')
   ```

   위 예시에서 `test.mp3`의 경로는 `'openpibo_files/audio/test.mp3'` 입니다.

3. 미리 지정해둔 경로를 참조하는 방법

   config.json 파일에 미리 지정해둔 경로를 호출하여 사용할 수 있습니다.

   기본적으로 config.json에는 'DATA_PATH' key의 value로 '/home/pi/openpibo_files' 경로가 설정되어있습니다.

   python에서 config.json을 호출하는 방법은 다음과 같습니다.

   ```python
   import openpibo

   print(openpibo.config)
   # {'DATA_PATH': '/home/pi/openpibo-files', 'KAKAO_ACCOUNT': '*****', 'robotId': ''}
   ```

   이를 활용하여 test.mp3 파일의 경로를 다음과 같이 입력할 수 있습니다.

   ```python
   pibo_audio.play(openpibo.config['DATA_PATH']+'/audio/test.mp3')
   ```

   위 예시에서 `test.mp3`의 경로는 `openpibo.config['DATA_PATH']+'/audio/test.mp3'` 입니다.

   *[참고] config.json 파일을 수정하여 데이터 경로를 커스텀하여 사용할 수도 있습니다.*

   1. 데이터를 저장할 디렉터리를 생성하고 파일을 저장합니다.

      - /home/pi/ 경로에 mydata 디렉터리를 생성합니다.
      ![](images/usage/mkdir_mydata_ad.png)

      - 생성한 디렉터리에 test.mp3 파일을 복사합니다.
      ![](images/usage/cp_test.png)

   2. config.json 파일을 수정합니다.

      ```bash
      $ sudo vi /home/pi/config.json # config.json 파일 에디터를 실행합니다.
      ```

      ![](images/usage/config.png)

      - i 키를 누르면 끼워넣기 모드가 활성화됩니다. 이때 키보드 타이핑을 하면 글씨가 입력됩니다.

      ![](images/usage/config_mydata.png)

      - 입력을 마치면, esc 키를 눌러 끼워넣기 모드를 종료합니다. 이후 `:wq` 를 타이핑하여 파일을 저장하고 종료합니다.

      ![](images/usage/wq.png)

   3. 경로를 호출해서 사용합니다.

      ```python
      import openpibo

      pibo_audio.play(openpibo.config['MY_DATA_PATH']+'/test.mp3')
      ```

      위 예시에서 `test.mp3`의 경로는 `openpibo.config['MY_DATA_PATH']+'/test.mp3'` 입니다.
