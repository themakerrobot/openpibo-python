"""
메인 컨트롤러를 제어합니다.

Class:
:obj:`~openpibo.device.Device`

메인 컨트롤러로 메시지 (``#code:data!`` 포맷) 가 전달됩니다.
이후 메인컨트롤러는 메시지를 파악하여 주어진 기능을 수행합니다.

다음 메시지 상세 설명은 기능 이름에 대해 code, data, get, set으로 설명됩니다.
**code** 와 **data** 는 메인 컨트롤러로 전송되는 메시지이며,
**get** 은 메시지 전달 후 반환 값, **set** 은 메시지 전달 후 수행되는 디바이스 설정을 의미합니다.

:메시지 상세 설명:

  * VERSION

    파이보의 버전을 출력합니다.

    * code: 10
    * data: -
    * get: 버전정보 (예: '10:FWN200312A')
    * set: -

  * HALT

    파이보를 종료합니다.

    * code: 11
    * data: -
    * get: 'ok'
    * set: 전원종료 요청

  * DC_CONN

    파이보의 충전기 연결 여부를 확인합니다.

    * code: 14
    * data: -
    * get: DC잭 연결정보 (예: '14:on')

      * ``on``: 연결 되어있음
      * ``off``: 연결 되어있지 않음

    * set: -

  * BATTERY

    파이보의 배터리 잔량을 확인합니다.

    * code: 15
    * data: -
    * get: 배터리 잔량 정보 (예: '15:100%')
    * set: -

  * REBOOT

    파이보의 설정을 초기화합니다.

    * code: 17
    * data: -
    * get: 'ok'
    * set: device 설정 초기화 요청

  * NEOPIXEL

    파이보의 눈(네오픽셀) 색을 변경합니다.

    * code: 20
    * data: 네오픽셀 색 R,G,B ('R,G,B') (예: '255,255,255')

      * 'R,G,B' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수

    * get: 'ok'
    * set: 네오픽셀설정 (R,G,B) 양쪽 동일하게 설정

  * NEOPIXEL_FADE

    파이보의 눈(네오픽셀) 색을 천천히 변경합니다.

    * code: 21
    * data: 네오픽셀 색 R,G,B와 색 변경 속도 d ('R,G,B,d') (예: '255,255,255,10')

      * 'R,G,B,d' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수
      * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms

    * get: 'ok'
    * set: 네오픽셀설정 (R,G,B) 양쪽 동일하게 설정 (색상 천천히 변경)

  * NEOPIXEL_BRIGHTNESS

    파이보의 눈(네오픽셀) 밝기를 조절합니다.

    * code: 22
    * data: 네오픽셀 밝기 (예: 64)

      * 0~255 정수
      * 기본값: 64

    * get: 'ok'
    * set: 네오픽셀설정 밝기를 설정한다. (기본: 64)

  * NEOPIXEL_EACH

    파이보의 양쪽 눈(네오픽셀) 색을 각각 변경합니다.

    * code: 23
    * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')

      * 'R,G,B,R,G,B' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수

    * get: 'ok'
    * set: 네오픽셀설정 (R,G,B,R,G,B) 양쪽 각각 설정

  * NEOPIXEL_FADE_EACH

    파이보의 양쪽 눈(네오픽셀) 색을 각각 천천히 변경합니다.

    * code: 24
    * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B와 색 변화 속도 d ('R,G,B,R,G,B,d') (예: '255,255,255,255,255,255,10')

      * 'R,G,B,R,G,B' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수
      * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms

    * get: 'ok'
    * set: 네오픽셀설정 (R,G,B,R,G,B) 양쪽 각각 설정 (색상 천천히 변경)

  * NEOPIXEL_LOOP

    파이보의 눈(네오픽셀) 색을 무지개색으로 일정시간동안 변경합니다.

    * code: 25
    * data: 색 변화 속도 d (예: 10)

      * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms

    * get: 'ok'
    * set: 네오픽셀을 무지개색으로 일정시간 반복

  * NEOPIXEL_OFFSET_SET

    파이보의 눈(네오픽셀) 색의 설정 시 반영 정도를 설정합니다.
    만약 오프셋이 '255,255,255,255,255,0' 이면, 네오픽셀을 흰 색으로 설정해도 오른쪽 눈에서 파란빛이 나오지 않습니다.

    * code: 26
    * data: 왼쪽 네오픽셀 오프셋 R,G,B와 오른쪽 네오픽셀 오프셋 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')

      * 'R,G,B,R,G,B' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수

    * get: 'ok'
    * set: 네오픽셀 최댓값 설정 (255,255,0,255,255,0 이면, white로 설정해도 노란색으로 표현됨.)

  * NEOPIXEL_OFFSET_GET

    파이보의 눈(네오픽셀) 오프셋 정보를 반환합니다.

    * code: 27
    * data: -
    * get: 네오픽셀 오프셋 정보 (예: '27:255,255,255,255,255,255')
    * set: -

  * NEOPIXEL_EACH_ORG

    파이보의 양쪽 눈(네오픽셀) 색을 각각 변경합니다. 단, 오프셋의 영향을 받지 않습니다.

    * code: 28
    * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')

      * 'R,G,B,R,G,B' 의 포맷으로 입력
      * 각 R, G, B는 0~255 정수

    * get: 'ok'
    * set: 네오픽셀설정 (R,G,B,R,G,B) 양쪽 각각 설정 (OFFSET의 영향을 받지 않음)

  * PIR

    PIR 센서를 켜고 끕니다.

    * code: 30
    * data: PIR 센서의 활성화 여부 (예: 'on')

      * ``on`` : PIR 센서를 켭니다.
      * ``off`` : PIR 센서를 끕니다.

    * get: 'ok'
    * set: PIR 센서 활성화/비활성화

  * SYSTEM

    각종 시스템 정보를 출력합니다.

    * code: 40
    * data: -
    * get: 아래의 정보가 모두 출력됩니다.

      * PIR 감지:

        * ``person`` : PIR센서가 적외선의 변화를 감지할 때 출력됩니다.
        * ``nobody`` : ``person`` 출력 후 2초간 적외선의 변화가 없을 때 1회 출력됩니다.
        * ``-`` : 적외선의 변화가 감지되지 않을 때 또는 PIR센서가 비활성화 상태일 때 출력됩니다.

      * Touch 감지:

        * ``touch`` : 터치센서 감지 시 출력됩니다.
        * ``-`` : 터치센서 감지가 안되면 출력됩니다.

      * DC잭 연결감지:

        * ``on`` : DC잭 감지 시 1회 출력됩니다. 이후 ``-`` 가 출력됩니다.
        * ``off`` : DC잭 감지 해제시 1회 출력됩니다. 이후 ``-`` 가 출력됩니다.
        * ``-`` : DC잭 신호의 변화가 없을 때 출력됩니다.

      * 버튼 감지:

        * ``on`` : 전원 버튼 누름 감지 시 출력됩니다.
        * ``-`` : 전원 버튼 누름 감지가 안되면 출력됩니다.

      * 시스템리셋: 현재 지원되지 않습니다.
      * 전원종료: 현재 지원되지 않습니다.
      
    * set: -

"""

import serial
import time
from threading import Lock

class Device:
  """
Functions:
:meth:`~openpibo.device.Device.locked`
:meth:`~openpibo.device.Device.send_cmd`
:meth:`~openpibo.device.Device.send_raw`

  메인컨트롤러를 제어하여 파이보의 여러가지 상태를 체크하거나, 눈 색깔을 변경합니다.

  example::

    from openpibo.device import Device

    pibo_device = Device()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """
  code_list ={
    "VERSION"               :"10",
    "HALT"                  :"11",
    "DC_CONN"               :"14",
    "BATTERY"               :"15",
    "REBOOT"                :"17",
    "NEOPIXEL"              :"20",
    "NEOPIXEL_FADE"         :"21",
    "NEOPIXEL_BRIGHTNESS"   :"22",
    "NEOPIXEL_EACH"         :"23",
    "NEOPIXEL_FADE_EACH"    :"24",
    "NEOPIXEL_LOOP"         :"25",
    "NEOPIXEL_OFFSET_SET"   :"26",
    "NEOPIXEL_OFFSET_GET"   :"27",
    "NEOPIXEL_EACH_ORG"     :"28",
    "PIR"                   :"30",
    "SYSTEM"                :"40",
  }
  """
  Device가 사용하는 ``code_list`` 정보입니다.
  """

  def __init__(self):
    """
    Device 클래스를 초기화합니다.
    """

    self.dev = serial.Serial(port="/dev/ttyS0", baudrate=9600)
    self.lock = Lock()
    self.code_val_list = [i for i in Device.code_list.values()]

  def locked(self):
    """
    Device가 사용 중인지 확인합니다. Device가 사용 중일 때 메시지를 보내지 않도록 주의합니다.

    example::

      pibo_device.locked()

    :returns: ``True`` / ``False``
    """

    return self.lock.locked()

  def send_cmd(self, code, data:str=""):
    """
    Device에 메시지 코드/데이터를 전송합니다. 입력된 메시지는 ``#code:data!`` 의 포맷으로 변경되어 전달됩니다.
    메시지 상세 설명은 위에서 확인할 수 있습니다.

    example::

      pibo_device.send_cmd(20, '255,255,255')

    :param str or int code: 메시지 코드

      * 10 : VERSION
      * 11 : HALT
      * 14 : DC_CONN
      * 15 : BATTERY
      * 17 : REBOOT
      * 20 : NEOPIXEL
      * 21 : NEOPIXEL_FADE
      * 22 : NEOPIXEL_BRIGHTNESS
      * 23 : NEOPIXEL_EACH
      * 24 : NEOPIXEL_FADE_EACH
      * 25 : NEOPIXEL_LOOP
      * 26 : NEOPIXEL_OFFSET_SET
      * 27 : NEOPIXEL_OFFSET_GET
      * 28 : NEOPIXEL_EACH_ORG
      * 30 : PIR
      * 40 : SYSTEM

    :param str data: 메시지

      ``code`` 의 값에 따라 데이터의 형식이 다릅니다. code에 따라 data가 요구되지 않을 수도 있습니다.

      example::

        pibo_device.send_cmd(20, '255,255,255')
        pibo_device.send_cmd(21, '255,255,255,10')
        pibo_device.send_cmd(22, '64')
        pibo_device.send_cmd(23, '255,255,255,255,255,255')
        pibo_device.send_cmd(24, '255,255,255,255,255,255,10')
        pibo_device.send_cmd(25, '2')
        pibo_device.send_cmd(26, '255,255,255,255,255,255')
        pibo_device.send_cmd(28, '255,255,255,255,255,255')
        pibo_device.send_cmd(30, 'on')

      **자세한 설명은 상단 "메시지 상세 설명" 참고하시기 바랍니다**

    :returns str: Device로부터 받은 응답

    """

    if not str(code) in self.code_val_list:
      raise Exception(f'"{code}" not support')

    return self.send_raw("#{}:{}!".format(code, data))

  def send_raw(self, raw):
    """
    Device에 실제 메시지를 전송하고 응답을 받습니다.
    메시지 상세 설명은 위에서 확인할 수 있습니다.

    example::

      pibo_device.send_raw('#20:255,255,255!')
      pibo_device.send_raw('#22:64!')

    :param str raw: 실제 전달되는 메시지

      Device에 전송되는 메시지의 포맷은 ``#code:data!`` 입니다.
      해당 메시지를 메인 컨트롤러에 전송합니다.

      **자세한 설명은 상단 "메시지 상세 설명" 참고하시기 바랍니다**

    :returns: Device로부터 받은 응답
    """

    #if self.locked() == True:
    #  return False

    self.lock.acquire()
    self.dev.write(raw.encode('utf-8'))
    data = ""
    time.sleep(0.05)

    while True:
      ch = self.dev.read().decode()
      if ch == '#' or ch == '\r' or ch == '\n':
        continue
      if ch == '!':
        break
      data += ch

    self.lock.release()
    return data

