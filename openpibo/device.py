"""
MCU를 제어하여, 부품을 제어합니다.

Class:
:obj:`~openpibo.device.Device`
"""

import serial
import time
from threading import Lock
import requests
import urllib

class Device:
  """
Functions:
:data:`~openpibo.device.Device.code_list`
:meth:`~openpibo.device.Device.send_cmd`
:meth:`~openpibo.device.Device.send_raw`
:meth:`~openpibo.device.Device.eye_on`
:meth:`~openpibo.device.Device.eye_on_s`
:meth:`~openpibo.device.Device.eye_off`
:meth:`~openpibo.device.Device.get_battery`
:meth:`~openpibo.device.Device.get_dc`
:meth:`~openpibo.device.Device.get_system`
:meth:`~openpibo.device.Device.get_pir`
:meth:`~openpibo.device.Device.get_touch`
:meth:`~openpibo.device.Device.get_button`

  메인컨트롤러를 제어하여 파이보의 여러가지 상태를 체크하거나, 눈 색깔을 변경합니다.

  example::

    from openpibo.device import Device

    device = Device()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.

  메시지 상세 설명::

    * VERSION(10): 파이보의 버전을 출력합니다.

      * msg: '#10:!'
      * result: 버전정보 (예: '10:FWN200312A')

    * HALT(11): 파이보를 종료합니다.

      * msg: '#11:!'
      * result: 'ok'

    * DC_CONN(14): 파이보의 충전기 연결 여부를 확인합니다.

      * msg: '#14:!'
      * result: '14:on'
        * ``on``: 연결 되어있음
        * ``off``: 연결 되어있지 않음

    * BATTERY(15): 파이보의 배터리 잔량을 확인합니다.

      * msg: '#15:!'
      * result: 배터리 잔량 정보 (예: '15:100%')

    * REBOOT(17): 설정 초기화 (Not use) 

      * msg: '#17:!'
      * result: 'ok'

    * NEOPIXEL(20): 파이보의 눈(네오픽셀) 색을 변경합니다. - 양쪽 동일하게 설정

      * msg: '#20:255,255,255!'
      * data: 네오픽셀 색 R,G,B ('R,G,B') (예: '255,255,255')
        * 'R,G,B' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
      * result: 'ok'

    * NEOPIXEL_FADE(21): 파이보의 눈(네오픽셀) 색을 천천히 변경합니다. - 양쪽 동일하게 설정

      * msg: '#21:255,255,255,10!'
      * data: 네오픽셀 색 R,G,B와 색 변경 속도 d ('R,G,B,d') (예: '255,255,255,10')
        * 'R,G,B,d' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
        * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms
      * result: 'ok'

    * NEOPIXEL_BRIGHTNESS(22): 파이보의 눈(네오픽셀) 밝기를 조절합니다.

      * msg: '#22:64!'
      * data: 네오픽셀 밝기 (예: 64)
        * 0~255 정수
        * 기본값: 64
      * result: 'ok'

    * NEOPIXEL_EACH(23): 파이보의 양쪽 눈(네오픽셀) 색을 각각 변경합니다. - 양쪽 각각 설정

      * msg: '#23:255,255,255,255,255,255!'
      * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')
        * 'R,G,B,R,G,B' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
      * result: 'ok'

    * NEOPIXEL_FADE_EACH(24): 파이보의 양쪽 눈(네오픽셀) 색을 각각 천천히 변경합니다.

      * msg: '#24:255,255,255,255,255,255,10!'
      * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B와 색 변화 속도 d ('R,G,B,R,G,B,d') (예: '255,255,255,255,255,255,10')
        * 'R,G,B,R,G,B' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
        * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms
      * result: 'ok'

    * NEOPIXEL_LOOP(25): 파이보의 눈(네오픽셀) 색을 무지개색으로 일정시간동안 변경합니다.

      * msg: '#25:10!'
      * data: 색 변화 속도 d (예: 10)
        * d는 색상 단위 변화 1당 걸리는 시간으로, 단위는 ms
      * result: 'ok'

    * NEOPIXEL_OFFSET_SET(26): 파이보의 눈(네오픽셀) 색의 초기 설정

      * msg: '#26:255,255,255,255,255,255!'
      * data: 왼쪽 네오픽셀 오프셋 R,G,B와 오른쪽 네오픽셀 오프셋 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')
        * 'R,G,B,R,G,B' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
      * result: 'ok'

    * NEOPIXEL_OFFSET_GET(27): 파이보의 눈(네오픽셀) 오프셋 정보를 반환합니다.

      * msg: '#27:255,255,255,255,255,255!'
      * data: 네오픽셀 오프셋 정보 (예: '27:255,255,255,255,255,255')
      * result: -

    * NEOPIXEL_EACH_ORG(28): 파이보의 양쪽 눈(네오픽셀) 색을 각각 변경합니다. 단, 오프셋의 영향을 받지 않습니다.

      * msg: '#28:255,255,255,255,255,255!'
      * data: 왼쪽 네오픽셀 색 R,G,B와 오른쪽 네오픽셀 색 R,G,B ('R,G,B,R,G,B') (예: '255,255,255,255,255,255')
        * 'R,G,B,R,G,B' 의 포맷으로 입력
        * 각 R, G, B는 0~255 정수
      * result: 'ok'

    * PIR(30): PIR 센서를 켜고 끕니다.

      * msg: '#30:on!'
      * data: PIR 센서의 활성화 여부 (예: 'on')
        * ``on`` : PIR 센서를 켭니다.
        * ``off`` : PIR 센서를 끕니다.
      * result: 'ok'

    * SYSTEM(40): 각종 시스템 정보를 출력합니다.

      * msg: '#40:!'
      * data: -
      * result: 아래의 정보가 모두 출력됩니다.
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
   
  def __init__(self, api_mode=True, port=80):
    """
    Device 클래스를 초기화합니다.
    """

    if api_mode == False:
      self.dev = serial.Serial(port="/dev/ttyS0", baudrate=9600)
      self.lock = Lock()

    self.api_mode = api_mode
    self.port = port
    self.code_val_list = [i for i in Device.code_list.values()]

  def send_cmd(self, code, data:str=""):
    """
    Device에 메시지 코드/데이터를 전송합니다. 입력된 메시지는 ``#code:data!`` 의 포맷으로 변경되어 전달됩니다.
    메시지 상세 설명은 위에서 확인할 수 있습니다.

    example::

      device.send_cmd(20, '255,255,255')

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

        device.send_cmd(20, '255,255,255')
        device.send_cmd(21, '255,255,255,10')
        device.send_cmd(22, '64')
        device.send_cmd(23, '255,255,255,255,255,255')
        device.send_cmd(24, '255,255,255,255,255,255,10')
        device.send_cmd(25, '2')
        device.send_cmd(26, '255,255,255,255,255,255')
        device.send_cmd(28, '255,255,255,255,255,255')
        device.send_cmd(30, 'on')

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

      device.send_raw('#20:255,255,255!')
      device.send_raw('#22:64!')

    :param str raw: 실제 전달되는 메시지

      Device에 전송되는 메시지의 포맷은 ``#code:data!`` 입니다.
      해당 메시지를 메인 컨트롤러에 전송합니다.

      **자세한 설명은 상단 "메시지 상세 설명" 참고하시기 바랍니다**

    :returns: Device로부터 받은 응답
    """
    # if self.lock.locked() == True:
    #  return False

    if self.api_mode == True: # RESTAPI 모드
      return requests.get(f"http://0.0.0.0:{self.port}/device/{urllib.parse.quote(raw)}").json()

    self.lock.acquire()
    self.dev.write(raw.encode('utf-8'))

    data = ""
    while True:
      ch = self.dev.read().decode()
      if ch == '#' or ch == '\r' or ch == '\n':
        continue
      if ch == '!':
        break
      data += ch

    time.sleep(0.01)
    self.lock.release()
    return data

  def eye_on(self, *color, intv=0):
    """
    LED를 켭니다.

    example::

      device.eye_on(255,0,0)	# 양쪽 눈 제어
      device.eye_on(0,255,0,0,0,255) # 양쪽 눈 각각 제어

    :param color:

      * RGB (0~255 숫자)
      * (R, G, B) -> 양쪽 눈 함께 제어
      * (R,G,B,R,G,B) -> 양쪽 눈 각각 제어

    :param intv:

      * interval (RGB 값이 1씩 바뀌는 시간)
      * 0 이면 off

    """

    if len(color) == 0:
      raise Exception("color is required")

    if len(color) not in (3, 6):
      raise Exception(f"len({color}) must be 3 or 6")

    for v in color:
      if v < 0 or v > 255:
        raise Exception("All color must be 0~255")

    if intv == 0:
      code = '20' if len(color) == 3 else '23'
      return self.send_raw(f'#{code}:{",".join(str(p) for p in color)}!')
    else:
      code = '21' if len(color) == 3 else '24'
      return self.send_raw(f'#{code}:{",".join(str(p) for p in color)},{intv}!')

  def eye_on_s(self, colors, intv=0):
    """
    LED를 켭니다.

    example::

      device.eye_on_s(['#ffffff', '#ffffff'])	# 양쪽 눈 제어

    :param colors:

      * RGB 16진수 문자 리스트 ['#ffffff', '#ffffff']

    """

    data = []
    data.append(int(colors[0][1:3], 16))
    data.append(int(colors[0][3:5], 16))
    data.append(int(colors[0][5:7], 16))
    data.append(int(colors[1][1:3], 16))
    data.append(int(colors[1][3:5], 16))
    data.append(int(colors[1][5:7], 16))

    if intv == 0:
      return self.send_raw(f'#23:{",".join(str(p) for p in data)}!')
    else:
      return self.send_raw(f'#24:{",".join(str(p) for p in data)},{intv}!')

  def eye_off(self):
    """
    LED를 끕니다.

    example::

      device.eye_off()

    """

    return self.send_raw('#20:0,0,0!')

  def get_battery(self, v=False):
    """
    배터리 정보를 요청합니다.

    example::

      device.get_battery()

    :param v: 값만 가져올지 메시지 전체를 가져올지 선택

    :returns: 배터리 정보 응답
    """


    return self.send_raw('#15:!') if v == False else self.send_raw('#15:!').split(':')[1]
  
  def get_dc(self, v=False):
    """
    DC커넥터 정보를 요청합니다.

    example::

      device.get_dc()

    :param v: 값만 가져올지 메시지 전체를 가져올지 선택

    :returns: DC커넥터 연결 정보 응답
    """

    return self.send_raw('#14:!') if v == False else self.send_raw('#14:!').split(':')[1]

  def get_system(self, v=False):
    """
    시스템 메시지를 요청합니다.

    example::

      device.get_system()

    :param v: 값만 가져올지 메시지 전체를 가져올지 선택

    :param name: 가져올 값 선택 ('pir'|'touch'|'dc'|'button'|'all')

    :returns: 시스템 메시지 정보 응답 (pir-touch-dc-button) 
    """

    return self.send_raw('#40:!') if v == False else self.send_raw('#40:!').split(':')[1]

  def get_pir(self):
    """
    시스템 메시지를 요청합니다. (pir)

    example::

      device.get_pir()

    :returns: 시스템 메시지 정보 응답에서 pir 값 추출
    """

    return self.send_raw('#40:!').split(':')[1].split('-')[0]

  def get_touch(self):
    """
    시스템 메시지를 요청합니다. (touch)

    example::

      device.get_touch()

    :returns: 시스템 메시지 정보 응답에서 touch 값 추출
    """

    return self.send_raw('#40:!').split(':')[1].split('-')[1]

  def get_button(self):
    """
    시스템 메시지를 요청합니다. (button)

    example::

      device.get_button()

    :returns: 시스템 메시지 정보 응답에서 button 값 추출
    """

    return self.send_raw('#40:!').split(':')[1].split('-')[3]

if __name__ == "__main__":
  device = Device()

  print(1, device.eye_on(255,0,0))
  print(2, device.send_raw('#21:0,0,255,10'))
  print(3, device.get_battery())
  print(4, device.get_dc())
  print(5, device.get_system())
  print(6, device.send_raw('#40:!'))
