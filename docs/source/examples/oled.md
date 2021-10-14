# oled

## figure_test.py

OLED display에 도형과 선을 출력합니다.

```python
from openpibo.oled import Oled

def run():
  o = Oled()
  o.clear()                              # 화면 지우기
  o.draw_rectangle((10,10,30,30), True)  # 길이가 20인 채워진 사각형 그리기
  o.draw_ellipse((70,40,90,60), False)   # 지름이 20인 빈 원 그리기
  o.draw_line((15,15,80,50))             # 선 그리기
  o.show()                               # 화면에 표시

if __name__ == "__main__":
  run()
```

**figure_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/oled $ sudo python3 figure_test.py 
```

**figure_test.py 결과**

![](images/oled_figure_test.jpg)

## image_test.py

OLED display에 이미지를 출력합니다. (128X64만 가능합니다.)

```python
import time

import openpibo
from openpibo.oled import Oled

# 화면에 clear.png 이미지 5초간 표시
def run():
  o = Oled()
  o.draw_image(openpibo.config['DATA_PATH']+"/image/clear.png")  # clear.png 그리기
  o.show()   # 화면에 표시
  time.sleep(5) # 5초동안 프로세스 정지
  o.clear()  # 화면 지우기
  o.show()

if __name__ == "__main__":
  run()
```

**image_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/oled $ sudo python3 image_test.py
```

**image_test.py 결과**

![](images/oled_image.png)

## self_test.py

직접 실습해보세요.

```python
from openpibo.oled import Oled

def run():
  """
  make your own code
  """


if __name__ == "__main__":
  o = Oled()

  run()
```

**self_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/oled $ sudo python3 self_test.py
```

## text_test.py

OLED display에 문자열을 출력합니다.

```python
from openpibo.oled import Oled

# (0,0), (0,20)에 15 크기의 text 표시
def run():
  o = Oled()
  o.set_font(size=15)
  o.draw_text((0, 0), "안녕? 난 파이보야 ")  # (0,0)에 문자열 출력
  o.draw_text((0,20), "☆  ★ ") # (0,20)에 문자열 출력
  o.show() # 화면에 표시

"""
  for count in range(5):
    oObj.clear()
    oObj.draw_text((10,10), "Hello World:{}".format(count))
    oObj.show()
    time.sleep(1)

  oObj.clear()
"""

if __name__ == "__main__":
  run()
```

**text_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/oled $ sudo python3 text_test.py
```

**text_test.py 결과**

![](images/oled_text_test.jpg)
