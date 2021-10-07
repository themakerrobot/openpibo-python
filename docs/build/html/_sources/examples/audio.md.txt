# audio

## play_test.py

mp3 파일을 재생 및 정지합니다.

```python
import time

import openpibo
from openpibo.audio import Audio

# test.mp3 파일 5초 재생 후 정지
def tts_f():
  obj = Audio()
  obj.play(filename=openpibo.config['DATA_PATH']+"/audio/test.mp3", out='local', volume=-2000)
  time.sleep(5) # 5초동안 프로세스 정지
  obj.stop()

if __name__ == "__main__":
  tts_f()

```

**play_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-example/audio $ sudo python3 play_test.py
```

**play_test.py 결과**

음악이 5초간 재생됩니다.