# 소프트웨어

교육용 파이보와 **openpibo-python** 패키지에 대한 가이드를 제공합니다.
[Github](https://github.com/themakerrobot)

## python package 구성

파이보의 다양한 기능을 사용할 수 있는 Class가 저장된 파일입니다.

```
openpibo
├── audio.py
├── collect.py
├── device.py
├── motion.py
├── oled.py
├── speech.py
└── vision.py
```

세부 가이드는 아래의 링크를 참조해주세요.

- [audio.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/audio.html)
- [collect.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/collect.html)
- [device.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/device.html)
- [motion.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/motion.html)
- [oled.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/oled.html)
- [speech.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/speech.html)
- [vision.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/vision.html)

## python 코드 작성

audio 라이브러리를 통해, openpibo 패키지를 사용하는 방법을 설명합니다.

```python
from openpibo.<라이브러리 명> import <클래스 명>

<인스턴스 명> = <클래스 명>()
<인스턴스 명>.<메소드명>(<인자>)
```

예) audio 라이브러리의 Audio 클래스의 함수를 사용하는 방법은 다음과 같습니다.

```python
from openpibo.audio import Audio

pibo_audio = Audio()

# play 메소드: 오디오 파일을 재생합니다.
pibo_audio.play('/home/pi/openpibo_files/audio/test.mp3')

# stop 메소드: 재생 중인 오디오 파일을 중지합니다.
pibo_audio.stop()

# mute 메소드: 음소거 모드로 전환합니다.
pibo_audio.mute(True)

# pibo_audio 는 Audio 클래스의 인스턴스
```

좌측의 LIBRARIES 탭을 참고하시기 바랍니다.

