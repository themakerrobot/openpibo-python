# motion

## motion_test.py

```python
from openpibo.motion import Motion

# wave3 모션 10번 반복
def run():
  o = Motion()
  o.set_motion(name="wave3", cycle=10)

if __name__ == "__main__":
  run()
```

아래는 motion_db.json의 일부입니다. motion_db는 `Motion Class의 get_motion` 메소드로 확인할 수 있습니다.

```
{
  ...
  "wave3": {
    "comment":"wave",
    "init_def":1,
    "init":[0,0,0,-25,0,0,0,0,0,25],
    "pos":[
      { "d": [   0,   0,   0,  25,   0,   0,  20,   0,   0,  25 ] , "seq": 450 },
      { "d": [ -20, 999, 999, 999, 999, 999, 999, 999, 999, -25 ] , "seq": 900 },
      { "d": [ 999, 999, 999, -25,  20, 999,   0, 999, 999, 999 ] , "seq": 1350 },
      { "d": [   0, 999, 999, 999, 999, 999, 999, 999, 999,  25 ] , "seq": 1800 },
      { "d": [ -20, 999, 999, 999, 999, 999,   0, 999, 999, -25 ] , "seq": 2250 },
      { "d": [ 999, 999, 999,  25, 999, 999,  20, 999, 999, 999 ] , "seq": 2700 },
      { "d": [   0, 999, 999, 999, -20, 999, 999, 999, 999,  25 ] , "seq": 3150 },
      { "d": [ 999, 999, 999, -25, 999, 999,   0, 999, 999, 999 ] , "seq": 3600 }
    ]
  },
  ...
}
```

**motion_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/motion $ sudo python3 motion_test.py
```

**motion_test.py 결과**

파이보가 `wave3`의 동작을 10회 반복합니다.

## motor_test.py

```python
import time

from openpibo.motion import Motion

def move(n, p, s, a):
  o.set_speed(n, s)         # n번 모터의 속도를 s로 변경
  o.set_acceleration(n, a)  # n번 모터의 가속도를 a로 변경
  o.set_motor(n, p)         # n번 모터의 위치를 p로 이동

# 'move() 실행 -> 3초 휴식 -> move() 실행 -> 3초 휴식'을 무한 반복
def run():
  while True:
    move(2, 30, 100, 10)
    move(8, 30,  10, 10)
    time.sleep(3)               # 단위: 초(sec)

    move(2, -30, 100, 10)
    move(8, -30,  10, 10)
    time.sleep(3)

if __name__ == "__main__":
  o = Motion()

  run()
```

**motor_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/motion $ sudo python3 motor_test.py 
```

**motor_test.py 결과**

파이보가 양팔을 서로 다른 속도로 무한히 움직입니다.

## multi_motor_test.py

```python
import time

from openpibo.motion import Motion

# 'set_motors() 실행 -> 1.1초 휴식 -> set_motors() 실행 -> 1.1초 휴식'을 무한 반복
def run():
  o = Motion()

  while True:
    o.set_motors(positions=[0,0,30,20, 30,0, 0,0,30,20], movetime=1000)
    time.sleep(1.1)
    o.set_motors(positions=[0,0,-30,-20, -30,0, 0,0,-30,-20], movetime=1000)
    time.sleep(1.1)

if __name__ == "__main__":
  run()
```

**multi_motor_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/motion $ sudo python3 multi_motor_test.py
```

**multi_motor_test.py 결과**

파이보가 양팔을 무한히 움직입니다.

## pymotor_test.py

```python
import time

from openpibo.motion import PyMotion

def move(n, p, s, a):
  o.set_speed(n, s)         # n번 모터의 속도를 s로 변경
  o.set_acceleration(n, a)  # n번 모터의 가속도를 a로 변경
  o.set_motor(n, p)         # n번 모터의 위치를 p로 이동

# 2초 간격으로 move() 실행 무한 반복
def run():
  while True:
    move(2, 30, 50,  0)
    move(8, 30, 50, 10)
    time.sleep(2)
  
    move(2,-30, 50,  0)
    move(8,-30, 50, 10)
    time.sleep(2)

if __name__ == "__main__":
  o = PyMotion()

  run()
```

**pymotor_test.py 실행**

```shell
pi@raspberrypi:~/openpibo-examples/motion $ sudo python3 pymotor_test.py 
```

**pymotor_test.py 결과**

파이보가 양팔을 무한히 움직입니다.
