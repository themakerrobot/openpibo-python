"""
각 메소드의 반환값은 다음과 같은 형식으로 구성됩니다.

* 실행 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": data 또는 None}``

  * 메소드에서 반환되는 데이터가 있을 경우 해당 데이터가 출력되고, 없으면 None이 출력됩니다.

* 실행 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

  * ``errcode`` 에 err 숫자 코드가, ``errmsg`` 에 해당 error 발생 원인이 출력됩니다.
  * err 숫자코드의 의미와 발생 원인은 다음과 같습니다.

    *  ``0`` : 메소드 실행 성공
    * ``-1`` : Argument error - 메소드 실행에 필요한 필수 인자 값 오류
    * ``-2`` : NotFound error - 존재하지 않는 파일 입력
    * ``-3`` : Runtime error - 동작 중 오류 발생
    * ``-4`` : Exception error - 위 error 이외의 다른 이유로 메소드 실행에 실패한 경우
"""

import os, sys, time, pickle

from .audio import Audio
from .collect import Wikipedia, Weather, News
from .oled import Oled
from .speech import Speech, Dialog
from .device import Device
from .motion import Motion
from .vision import Camera, Face, Detect
from .modules.vision.stream import VideoStream

from threading import Thread, Lock
from queue import Queue

"""
Class:
:obj:`~openpibo.edu_v1.Pibo`
"""

class Pibo:
  """
Functions:
:meth:`~openpibo.edu_v1.Pibo.play_audio`
:meth:`~openpibo.edu_v1.Pibo.stop_audio`
:meth:`~openpibo.edu_v1.Pibo.search_wikipedia`
:meth:`~openpibo.edu_v1.Pibo.search_weather`
:meth:`~openpibo.edu_v1.Pibo.search_news`
:meth:`~openpibo.edu_v1.Pibo.eye_on`
:meth:`~openpibo.edu_v1.Pibo.eye_off`
:meth:`~openpibo.edu_v1.Pibo.check_device`
:meth:`~openpibo.edu_v1.Pibo.thread_device`
:meth:`~openpibo.edu_v1.Pibo.start_thread_device`
:meth:`~openpibo.edu_v1.Pibo.stop_thread_device`
:meth:`~openpibo.edu_v1.Pibo.motor`
:meth:`~openpibo.edu_v1.Pibo.motors`
:meth:`~openpibo.edu_v1.Pibo.motors_movetime`
:meth:`~openpibo.edu_v1.Pibo.get_motion`
:meth:`~openpibo.edu_v1.Pibo.set_motion`
:meth:`~openpibo.edu_v1.Pibo.show_display`
:meth:`~openpibo.edu_v1.Pibo.draw_text`
:meth:`~openpibo.edu_v1.Pibo.draw_image`
:meth:`~openpibo.edu_v1.Pibo.draw_figure`
:meth:`~openpibo.edu_v1.Pibo.invert`
:meth:`~openpibo.edu_v1.Pibo.clear_display`
:meth:`~openpibo.edu_v1.Pibo.translate`
:meth:`~openpibo.edu_v1.Pibo.tts`
:meth:`~openpibo.edu_v1.Pibo.stt`
:meth:`~openpibo.edu_v1.Pibo.conversation`
:meth:`~openpibo.edu_v1.Pibo.thread_camera`
:meth:`~openpibo.edu_v1.Pibo.start_thread_camera`
:meth:`~openpibo.edu_v1.Pibo.stop_thread_camera`
:meth:`~openpibo.edu_v1.Pibo.capture`
:meth:`~openpibo.edu_v1.Pibo.search_object`
:meth:`~openpibo.edu_v1.Pibo.search_qr`
:meth:`~openpibo.edu_v1.Pibo.search_text`
:meth:`~openpibo.edu_v1.Pibo.search_color`
:meth:`~openpibo.edu_v1.Pibo.detect_face`
:meth:`~openpibo.edu_v1.Pibo.search_face`
:meth:`~openpibo.edu_v1.Pibo.train_face`
:meth:`~openpibo.edu_v1.Pibo.delete_face`
:meth:`~openpibo.edu_v1.Pibo.get_facedb`
:meth:`~openpibo.edu_v1.Pibo.save_facedb`
:meth:`~openpibo.edu_v1.Pibo.init_facedb`
:meth:`~openpibo.edu_v1.Pibo.load_facedb`
:meth:`~openpibo.edu_v1.Pibo.get_image`
:meth:`~openpibo.edu_v1.Pibo.return_msg`

  ``openpibo`` 의 다양한 기능들을 한번에 사용할 수 있는 클래스 입니다.

  다음 클래스의 기능을 모두 사용할 수 있습니다.

  * Device
  * Audio
  * Wikipedia
  * Weather
  * News
  * Oled
  * Speech
  * Dialog
  * Motion
  * Camera
  * Face
  * Detect

  example::

    from openpibo.edu_v1 import Pibo

    pibo_edu_v1 = Pibo()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """
  return_msg_list = {
    "Success": 0,
    "Argument error": -1,
    "NotFound error": -2,
    "Runtime error": -3,
    "Exception error": -4,
  }
  """
  반환되는 ``errmsg`` 에 대한 ``errcode`` 입니다.
  """

  def __init__(self):
    self.camera_loop = False
    self.img = None
    self.device_loop = False
    self.flash = False
    self.device = Device()
    self.audio = Audio()
    self.wikipedia = Wikipedia()
    self.weather = Weather()
    self.news = News()
    self.oled = Oled()
    self.speech = Speech()
    self.dialog = Dialog()
    self.motion = Motion()
    self.camera = Camera()
    self.face = Face()
    self.detect = Detect()
    self.que = Queue()
    self.motor_range = [25,35,80,30,50,25,25,35,80,30]
    self.device.send_cmd(Device.code_list['PIR'], "on")

  # [Audio] - Play mp3/wav files
  def play_audio(self, filename=None, out='local', volume='-2000', background=True):
    """
    입력한 경로의 파일을 재생합니다.

    example::

      pibo_edu_v1.play_audio('/home/pi/openpibo-files/data/audio/opening.mp3')

    :param str filename: 재생할 파일의 경로.

      ``mp3`` 와 ``wav`` 형식을 지원합니다.

    :param str out: 어느 포트에서 재생할지 선택합니다.

      ``local``, ``hdmi``, ``both`` 만 입력할 수 있습니다.

      (default: ``local``)

    :param str or int volume: 음량을 설정합니다.

      단위는 mdB 이고, 값이 커질수록 음량이 커집니다.

      음량이 매우 크므로 -2000 정도로 사용하는 것을 권장합니다.

      (default: ``-2000``)

    :param bool background: 오디오 파일을 백그라운드에서 실행할지 여부를 결정합니다.

      * ``True``: 오디오 재생 중에 다른 명령어를 사용할 수 있습니다. (default)
      * ``False``: 오디오 파일이 종료될 때 까지 다른 명령어를 실행할 수 없습니다.

    """

    try:
      if filename == None:
        return self.return_msg(False, "Argument error", "filename is required", None)
      if filename.split('.')[-1] not in ('mp3', 'wav'):
        return self.return_msg(False, "Argument error", f"{filename} must be (mp3|wav) file", None)
      if not os.path.isfile(filename):
        return self.return_msg(False, "NotFound error", f"{filename} does not exist", None)

      if out not in ('local', 'hdmi', 'both'):
        return self.return_msg(False, "Argument error", f"{out} must be (local|hdmi|both)", None)

      self.audio.play(filename, out, volume, background)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Audio] - Stop audio
  def stop_audio(self):
    """
    background에서 재생중인 오디오를 정지합니다.

    example::

      pibo_edu_v1.stop_audio()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.audio.stop()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Collect] - Wikipedia
  def search_wikipedia(self, topic=None):
    """
    위키백과에서 ``topic`` 를 검색합니다.

    example::

      pibo_edu_v1.search_wikipedia('강아지')

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": list 형태의 data}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

        대부분의 경우 '0'번 항목에 개요를 표시하고, 검색된 내용이 없을 경우 None을 반환합니다.

          example::

            ['0':{
              'title': '명칭', 
               'content': "한국어 ‘강아지’는 ‘개’에 어린 짐승을 뜻하는 ‘아지’가 붙은 말이다..."
            }, ... ]
            or
            None
    """

    try:
      if topic == None:
        return self.return_msg(False, "Argument error", "topic is required", None)
      result = self.wikipedia.search(topic)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Collect] - Weather
  def search_weather(self, region=None):
    """
    해당 지역(```region```)의 날씨 정보(종합예보, 오늘/내일/모레 날씨)를 가져옵니다.

    example::

      pibo_edu_v1.search_weather('전국')

    :param str search_region: 검색 가능한 지역 (default: 전국)

      검색할 수 있는 지역은 다음과 같습니다::

        '전국', '서울', '인천', '경기', '부산', '울산', '경남', '대구', '경북',
        '광주', '전남', '전북', '대전', '세종', '충남', '충북', '강원', '제주'

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": dict 형태의 data}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

      example::

        종합예보와 오늘/내일/모레의 날씨 및 최저/최고기온을 반환합니다.
        {
          'forecast': '내일 경기남부 가끔 비, 내일까지 바람 약간 강, 낮과 밤의 기온차 큼'
          'today':
          {
            'weather': '전국 대체로 흐림',
            'minimum_temp': '15.3 ~ 21.6',
            'highst_temp': '23.1 ~ 27.6'
          }
          'tomorrow':
          {
            'weather': '전국 대체로 흐림',
            'minimum_temp': '15.3 ~ 21.6', 
            'highst_temp': '23.1 ~ 27.6'
          }
          'after_tomorrow':
          {
            'weather': '전국 대체로 흐림',
            'minimum_temp': '15.3 ~ 21.6',
            'highst_temp': '23.1 ~ 27.6'
          }
        }
        or None
    """

    try:
      if region == None:
        return self.return_msg(False, "Argument error", "region is required", None)
      result = self.weather.search(region)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Collect] - News
  def search_news(self, topic=None):
    """
    해당 주제(```topic```)에 맞는 뉴스를 가져옵니다.

    example::

      pibo_edu_v1.search_weather('전국')

    :param str topic: 검색 가능한 뉴스 주제 (default: 뉴스랭킹)

      검색할 수 있는 주제는 다음과 같습니다::

        '속보', '정치', '경제', '사회', '국제', '문화', '연예', '스포츠',
        '풀영상', '뉴스랭킹', '뉴스룸', '아침&', '썰전 라이브', '정치부회의'

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": list 형태의 data}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

      example::

        [
          {
            'title': '또 소방차 막은 불법주차, 이번엔 가차없이 밀어버렸다', 
            'link': 'https://news.jtbc.joins.com/article/article.aspx?...',
            'description': '2019년 4월 소방당국의 불법주정차 강경대응 훈련 모습...,
            'pubDate': '2021.09.03'
          },  
        ]
        or None
    """

    try:
      if topic == None:
        return self.return_msg(False, "Argument error", "topic is required", None)
      result = self.news.search(topic)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Neopixel] - LED ON
  def eye_on(self, *color):
    """
    LED를 켭니다.

    example::

      pibo_edu_v1.eye_on(255,0,0)	# 양쪽 눈 제어
      pibo_edu_v1.eye_on(0,255,0,0,0,255) # 양쪽 눈 각각 제어

    :param color:

      * RGB (0~255 숫자)
      * (R, G, B) -> 양쪽 눈 함께 제어
      * (R,G,B,R,G,B) -> 양쪽 눈 각각 제어

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if len(color) == 0:
        return self.return_msg(False, "Argument error", "color is required", None)

      if len(color) not in (3, 6):
        return self.return_msg(False, "Argument error", f"len({color}) must be 3 or 6", None)

      for v in color:
        if v < 0 or v > 255:
          return self.return_msg(False, "Argument error", "All color must be 0~255", None)

      code = '20' if len(color) == 3 else '23'
      cmd = f'#{code}:{",".join(str(p) for p in color)}!'

      if self.device_loop:
        self.que.put(cmd)
      else:
        self.device.send_raw(cmd)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Neopixel] - LED OFF
  def eye_off(self):
    """
    LED를 끕니다.

    example::

      pibo_edu_v1.eye_off()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if self.device_loop:
        self.que.put('#20:0,0,0!')
      else:
        self.device.send_raw('#20:0,0,0!')
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Device] - Check device
  def check_device(self):
    """
    디바이스의 상태를 확인합니다. (일회성)

    example::

      pibo_edu_v1.check_device()

      BATTERY, PIR, TOUCH, DC_CONN, BUTTON의 상태를 조회할 수 있습니다.

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": Device로부터 응답}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.device.send_raw('#15:!')
      bat = result.split(':')[1]
      result = self.device.send_raw('#40:!')
      result = result.split(':')[1].split('-')
      ans = {"BATTERY":bat, "PIR": result[0], "TOUCH": result[1], "DC_CONN": result[2], "BUTTON": result[3]}
      return self.return_msg(True, "Success", "Success", ans)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Device] - thread_device
  def thread_device(self, func):
    """
    (``start_thread_device`` 의 내부함수)

    디바이스로 부터 기본 메시지를 수신하는 Thread 함수입니다.
    시스템체크(1초 주기), 배터리체크(10초 주기) 입니다.

    ``func`` 는 Device의 메시지를 수신합니다.
    """

    self.system_check_time = time.time()
    self.battery_check_time = time.time()

    while True:
      if self.device_loop == False:
        break

      if self.que.qsize():
        self.device.send_raw(self.que.get())

      if time.time() - self.system_check_time > 1:
        func(self.device.send_raw('#40:!'))
        self.system_check_time = time.time()

      if time.time() - self.battery_check_time > 10:
        func(self.device.send_raw('#15:!'))
        self.battery_check_time = time.time()
      time.sleep(0.01)


  # [Device] - Start thread_device
  def start_thread_device(self, func=None):
    """
    thread_device 함수를 시작합니다.

    example::

      def decode(msg):
        print(msg)

      pibo_edu_v1.start_thread_device(decode)

    :param func: thread_device 함수를 시작합니다.

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if func == None:
        return self.return_msg(False, "Argument error", "Func is required", None)
      if self.device_loop == True:
        return self.return_msg(False, "Runtime error", "thread_devices() is already running", None)

      self.device_loop = True
      t = Thread(target=self.thread_device, args=(func,))
      t.daemon = True
      t.start()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Device] - Stop thread_device
  def stop_thread_device(self):
    """
    디바이스의 상태 확인을 종료합니다.

    example::

      pibo_edu_v1.stop_thread_device()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.device_loop = False
      time.sleep(0.5)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Motion] - Control 1 motor(position/speed/accel)
  def motor(self, n=None, position=None, speed=None, accel=None):
    """
    모터 1개를 제어합니다.

    example::

      pibo_edu_v1.motor(2, 30, 100, 10)

    :param int n: 모터 번호 (0~9)

    :param int position: 모터 각도

      모터별 허용 각도 범위 절대값::

        [25,35,80,30,50,25,25,35,80,30]
        # 0번 모터는 -25 ~ 25 범위의 모터 각도를 가집니다.

    :param int speed: 모터 속도 (0~255)

      default: None - 사용자가 이전에 설정한 값으로 제어

    :param int accel: 모터 가속도 (0~255)

      default: None- 사용자가 이전에 설정한 값으로 제어

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if speed != None:
        if speed < 0 or speed > 255:
          return self.return_msg(False, "Argument error", f"{speed} must be 0~255", None)
        self.motion.set_speed(n, speed)
      if accel != None:
        if accel < 0 or accel > 255:
          return self.return_msg(False, "Argument error", f"{accel} must be 0~255", None)
        self.motion.set_acceleration(n, accel)

      if n == None:
        return self.return_msg(False, "Argument error", "n is required", None)
      if n < 0 or n > 9:
        return self.return_msg(False, "Argument error", f"{n} must be 0~9", None)

      if position == None:
        return self.return_msg(False, "Argument error", "position is required", None)
      if abs(position) > self.motor_range[n]:
        return self.return_msg(False, "Argument error", 
                f"Motor{n}'s position must be -{self.motor_range[n]}~{self.motor_range[n]}", None)

      self.motion.set_motor(n, position)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Motion] - Control all motors(position/speed/accel)
  def motors(self, positions=None, speeds=None, accels=None):
    """
    10개의 모터를 개별 제어합니다.

    example::

      pibo_edu_v1.motors(
        positions=[0,0,0,10,0,10,0,0,0,20],
        speed=[0,0,0,15,0,10,0,0,0,10],
        accel=[0,0,10,5,0,0,0,0,5,10]
      )

    :param list positions: 0-9번 모터 각도 배열

      모터별 허용 각도 범위 절대값::

        [25,35,80,30,50,25,25,35,80,30]
        # 0번 모터는 -25 ~ 25 범위의 모터 각도를 가집니다.

    :param list speed: 0-9번 모터 속도 (0~255)

      default: None - 사용자가 이전에 설정한 값으로 제어

    :param list accel: 0-9번 모터 가속도 (0~255)

      default: None - 사용자가 이전에 설정한 값으로 제어

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if speeds != None:
        for v in speeds:
          if v < 0 or v > 255:
            return self.return_msg(False, "Argument error", "All speed must be 0~255", None)
        self.motion.set_speeds(speeds)

      if accels != None:
        for v in accels:
          if v < 0 or v > 255:
            return self.return_msg(False, "Argument error", "All accel must be 0~255", None)
        self.motion.set_accelerations(accels)

      if len(positions) != 10:
        return self.return_msg(False, "Argument error", "positions are required", None)
      for n in range(len(positions)):
        if abs(positions[n]) > self.motor_range[n]:
          return self.return_msg(False, "Argument error", 
                  f"Motor{n}'s position must be -{self.motor_range[n]}~{self.motor_range[n]}", None)

      self.motion.set_motors(positions)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Motion] - Control all motors(movetime)
  def motors_movetime(self, positions=None, movetime=None):
    """
    입력한 시간 내에 모든 모터를 특정 위치로 이동합니다.

    example::

      pibo_edu_v1.motors_movetime(positions=[0,0,30,20, 30,0, 0,0,30,20], movetime=1000)
      # 1000ms 내에 모든 모터가 [0,0,30,20,30,0,0,0,30,20]의 위치로 이동

    :param list positions: 0-9번 모터 각도 배열

      모터별 허용 각도 범위 절대값::

        [25,35,80,30,50,25,25,35,80,30]
        # 0번 모터는 -25 ~ 25 범위의 모터 각도를 가집니다.

    :param int movetime: 모터 이동 시간(ms)

      모터가 정해진 위치까지 이동하는 시간

      * ``movetime`` 이 있으면 해당 시간까지 모터를 이동시키기 위한 속도, 가속도 값을 계산하여 모터를 제어합니다.
      * ``movetime`` 이 없으면 이전에 설정한 속도, 가속도 값에 의해 모터를 이동시킵니다.

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if len(positions) != 10:
        return self.return_msg(False, "Argument error", "positions are required", None)
      for n in range(len(positions)):
        if abs(positions[n]) > self.motor_range[n]:
          return self.return_msg(False, "Argument error", 
                  f"Motor{n}'s position must be -{self.motor_range[n]}~{self.motor_range[n]}", None)

      if movetime and movetime < 0:
        return self.return_msg(False, "Argument error", f"{movetime} must be positive number", None)

      self.motion.set_motors(positions, movetime)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Motion] - Get motion type or motion details
  def get_motion(self, name=None):
    """
    모션 종류 및 정보를 조회합니다.

    ``set_motion(name, cycle)`` 에서 사용할 name 값을 조회할 수 있습니다.

    ``get_motion()`` 으로 모션 목록을 조회한 후, 모션을 하나 선택하여 ``get_motion(name)`` 에서 해당 모션에 대한 상세 정보를 얻을 수 있습니다.

    example::

      pibo_edu_v1.get_motion()
      # ['stop', 'stop_body', 'sleep', 'lookup', 'left', ...]

      pibo_edu_v1.get_motion("sleep")
      # {'comment': 'sleep', 'init': [0,0,-70,-25,0,15,0,0,70,25], 'init_def': 0, ...}

    :param str name: 모션 이름

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": profile로부터 응답}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

    [전체 모션 리스트]::

      stop, stop_body, sleep, lookup, left, left_half, right, right_half, foward1-2, backward1-2, 
      step1-2, hifive, cheer1-3, wave1-6, think1-4, wake_up1-3, hey1-2, yes_h, no_h, breath1-3, 
      breath_long, head_h, spin_h, clapping1-2, hankshaking, bow, greeting, hand1-4, foot1-2, 
      speak1-2, speak_n1-2, speak_q, speak_r1-2, speak_l1-2, welcome, happy1-3, excite1-2, 
      boring1-2, sad1-3, handup_r, handup_l, look_r, look_l, dance1-5, motion_test, test1-4
      # foward1-2는 forward1, forward2 두 종류가 있음을 의미합니다.
    """

    try:
      ret = self.motion.get_motion(name)
      return self.return_msg(True, "Success", "Success", ret)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Motion] - Set motion
  def set_motion(self, name=None, cycle=1, profile_path=None):
    """
    모션의 동작을 실행합니다.

    example::

      pibo_edu_v1.set_motion("dance1", 5)

    :param str name: 모션 이름

    :param int cycle: 모션 반복 횟수

    :param str profile_path:

      커스텀 동작 프로파일 경로입니다. 입력하지 않으면 기본 프로파일을 불러옵니다.

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if name == None:
        return self.return_msg(False, "Argument error", "name is required", None)

      self.motion.set_motion(name, cycle, profile_path)
      return self.return_msg(ret, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Show display
  def show_display(self):
    """
    화면에 표시합니다.

    문자 또는 그림을 그린 후 이 메소드를 사용해야만 파이보의 oled에 표시가 됩니다.

    example::

      pibo_edu_v1.draw_text((10, 10), '안녕하세요', 10)
      pibo.show_display()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.oled.show()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Draw a text
  def draw_text(self, points=None, text=None, size=None):
    """
    문자를 씁니다. (한글/영어)

    `show_display` 메소드와 함께 사용하여 oled에 표시할 수 있습니다.

    example::

      pibo_edu_v1.draw_text((10, 10), '안녕하세요.', 15)
      pibo_edu_v1.show_display()

    :param tuple(int, int) points: 문자열의 좌측상단 좌표 튜플(x,y)

    :param str text: 문자열 내용

    :param int size: 폰트 크기

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if points == None or type(points) is not tuple or len(points) != 2:
        return self.return_msg(False, "Argument error", f"{points} must be (x, y)", None)

      if text == None or type(text) != str:
        return self.return_msg(False, "Argument error", "text is required", None)

      if size != None:
        self.oled.set_font(size=size)

      self.oled.draw_text(points, text)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Draw an image
  def draw_image(self, filename=None):
    """
    이미지를 그립니다. (128X64 png 파일)

    128X64 png 파일 외에는 지원하지 않습니다.

    `show_display` 메소드와 함께 사용하여 oled에 표시할 수 있습니다.

    example::

      pibo_edu_v1.draw_image("/home/pi/openpibo-files/data/image/clear.png")
      pibo_edu_v1.show_display()

    :param str filename: 이미지 파일의 경로

    :returns:

    * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
    * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if filename == None:
        return self.return_msg(False, "Argument error", "filename is required", None)
      if filename.split('.')[-1] != 'png':
        return self.return_msg(False, "Argument error", f"{filename} must be png file", None)
      if not os.path.isfile(filename):
        return self.return_msg(False, "NotFound error", f"{filename} does not exist", None)

      h, w = self.camera.imread(filename).shape[:2]
      if h != 64 or w != 128:
        return self.return_msg(False, "Runtime error", f"{filename}'s size must be 128X64", None)

      self.oled.draw_image(filename)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Draw a shpae
  def draw_figure(self, points=None, shape=None, fill=None):
    """
    도형을 그립니다. (사각형, 원, 선)

    `show_display` 메소드와 함께 사용하여 oled에 표시할 수 있습니다.

    example::

      pibo_edu_v1.draw_figure((10,10,30,30), "rectangle", True)
      pibo_edu_v1.draw_figure((70,40,90,60), "circle", False)
      pibo_edu_v1.draw_figure((15,15,80,50), "line")
      pibo_edu_v1.show_display()

    :param tuple(int, int, int, int) points: 선 - 시작 좌표, 끝 좌표(x1,y1,x2,y2)

      사각형, 원 - 좌측상단, 우측하단 좌표 튜플(x1,y1,x2,y2)

    :param str shape: 도형 종류 - ``rectangle`` / ``circle`` / ``line``

    :param bool fill: ``True`` (채움) / ``False`` (채우지 않음)

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if points == None or type(points) is not tuple or len(points) != 4:
        return self.return_msg(False, "Argument error", f"{points} must be (x1,y1,x2,y2)", None)

      if shape == None or type(shape) is not str:
        return self.return_msg(False, "Argument error", "shape is required", None)

      if shape == 'rectangle':
        self.oled.draw_rectangle(points, fill)
      elif shape == 'circle':
        self.oled.draw_ellipse(points, fill)
      elif shape == 'line':
        self.oled.draw_line(points)
      else:
        return self.return_msg(False, "Argument error", f"{shape} must be (rectangle|circle|line)", None)

      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Color inversion
  def invert(self):
    """
    이미지를 반전시킵니다. (색 반전)

    `show_display` 메소드와 함께 사용하여 oled에 표시할 수 있습니다.

    example::

      pibo_edu_v1.invert()
      pibo_edu_v1.show_display()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.oled.invert()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [OLED] - Clear display
  def clear_display(self):
    """
    OLED 화면을 지웁니다.

    example::

      pibo_edu_v1.clear_display()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.oled.clear()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Speech] - Sentence translation
  def translate(self, string=None, to='ko'):
    """
    구글 번역기를 이용해 문장을 번역합니다.

    example::

      pibo_edu_v1.translate('즐거운 금요일', 'en')

    :param str string: 번역할 문장

    :param str to: 번역할 언어(한글-ko / 영어-en)

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 번역된 문장}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if string == None:
        return self.return_msg(False, "Argument error", "string is required", None)
      if to not in ('ko', 'en'):
        return self.return_msg(False, "Argument error", f"{to} must be (ko|en)", None)

      result = self.speech.translate(string, to)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Speech] - TTS
  def tts(self, string, voice_type="WOMAN_READ_CALM", break_time=None, filename='tts.mp3'):
    """
    Text(문자)를 Speech(음성)로 변환합니다.

    example::

      pibo_edu_v1.tts("안녕하세요. 반갑습니다.", "WOMAN_READ_CALM", "/home/pi/tts.mp3")
      pibo_edu_v1.tts("안녕하세요. 반갑습니다.", "WOMAN_READ_CALM", 500, "/home/pi/tts.mp3")

    :param str string: 변환할 문장

    :param str voice_type: 목소리 종류

      * WOMAN_READ_CALM: 여성 차분한 낭독체 (default)
      * MAN_READ_CALM: 남성 차분한 낭독체
      * WOMAN_DIALOG_BRIGHT: 여성 밝은 대화체
      * MAN_DIALOG_BRIGHT: 남성 밝은 대화체

    :param int break_time: 쉬는 시간 (단위: ms)

    :param str filename: 저장할 파일 경로(mp3, wav)

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if string == None:
        return self.return_msg(False, "Argument error", "string is required", None)

      if filename.split('.')[1] != 'mp3':
        return self.return_msg(False, "Argument error", f"{filename} must be mp3 file", None)

      if voice_type not in ('WOMAN_READ_CALM', 'MAN_READ_CALM', 'WOMAN_DIALOG_BRIGHT', 'MAN_DIALOG_BRIGHT'):
        return self.return_msg(False, "Argument error", f"{voice_type} not support", None)

      if break_time != None:
        if type(break_time) is not int:
          return self.return_msg(False, "Argument error", f"{break_time} must be integer type", None)
        if break_time < 0:
          return self.return_msg(False, "Argument error", f"{break_time} must be positive number", None)
        string = f'<speak><voice name="{voice_type}">{string}<break time="{break_time}ms"/></voice></speak>' 
      else:
        string = f'<speak><voice name="{voice_type}">{string}</voice></speak>'

      print(string)
      self.speech.tts(string, filename)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Speech] - STT
  def stt(self, filename='stream.wav', timeout=5):
    """
    Speech(음성)를 Text(문자)로 변환합니다.

    example::

      pibo_edu_v1.stt('/home/pi/stream.wav', 5)

    :param str filename: 저장할 파일 경로

    :param int timeout: 녹음할 시간(s)

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 변환된 문장}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.speech.stt(filename, timeout)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Speech] - Conversation
  def conversation(self, q=None):
    """
    질문에 대한 답을 추출합니다.

    example::

      pibo_edu_v1.conversation('주말에 뭐하지?')
      # answer: 사탕 만들어요.

    :param str q: 질문

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 질문에 대한 응답}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if q == None:
        return self.return_msg(False, "Argument error", "q is required", None)
      if type(q) is not str:
        return self.return_msg(False, "Argument error", f"{q} must be str type", None)

      result = self.dialog.get_dialog(q)
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - start_thread_camera thread
  def thread_camera(self):
    """
    (``start_camera`` 메소드의 내부함수 입니다.)

    카메라로 짧은 주기로 사진을 찍어 128x64 크기로 변환한 후 OLED에 보여줍니다.
    """

    vs = VideoStream().start()

    while True:
      if self.camera_loop == False:
        vs.stop()
        break
      self.img = vs.read()
      img = self.img
      img = self.camera.convert_img(img, 128, 64)
      #_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
      if self.flash:
        img = self.camera.rotate(img, 10, 0.9)
        self.oled.draw_data(img)
        self.oled.show()
        time.sleep(0.3)
        self.flash = False
        continue
      self.oled.draw_data(img)
      self.oled.show()


  # [Vision] - Camera ON
  def start_thread_camera(self):
    """
    카메라가 촬영하는 영상을 OLED에 보여줍니다.

    example::

      pibo_edu_v1.start_thread_camera()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if self.camera_loop == True:
        return self.return_msg(False, "Runtime error", "thread_camera() is already running", None)

      self.camera_loop = True
      t = Thread(target=self.thread_camera, args=())
      t.daemon = True
      t.start()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Camera OFF
  def stop_thread_camera(self):
    """
    카메라를 종료합니다.

    example::

      pibo_edu_v1.stop_thread_camera()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.camera_loop = False
      time.sleep(0.5)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Capture
  def capture(self, filename="capture.png"):
    """
    사진을 촬영하여 이미지로 저장합니다.

    example::

      pibo_edu_v1.capture('/home/pi/test.png')

    :param str filename: 저장할 파일 경로

      이미지 파일 형식 기입 필수 - jpg, jpeg, png, bmp

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if filename.split('.')[1] not in ("png", "jpg", "jpeg", "bmp"):
        return self.return_msg(False, "Argument error", f"{filename} must be (png|jpg|jpeg|bmp)", None) 

      if self.camera_loop == True:
        self.camera.imwrite(filename, self.img)
        self.flash = True
      else:
        img = self.camera.read()
        self.camera.imwrite(filename, img)
        img = self.camera.convert_img(img, 128, 64)
        #_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.oled.draw_data(img)
        self.oled.show()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Detect object
  def search_object(self):
    """
    카메라 이미지 안의 객체를 인식합니다.

    example::

      pibo_edu_v1.search_object()

    인식 가능한 사물 목록::

      "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
      "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"

    :returns:

      * 성공:``{"result": True, "errcode": 0, "errmsg": "Success", "data": {"name": 이름, "score": 점수, "position": 사물좌표(startX, startY, endX, endY)}}``
      * 실패:``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.detect.detect_object(self.get_image())
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Detect QR/barcode
  def search_qr(self):
    """
    카메라 이미지 안의 QR 코드 및 바코드를 인식합니다.

    example::

      pibo_edu_v1.search_qr()

    :returns:

      * 성공:``{"result": True, "errcode": 0, "errmsg": "Success", "data": {"data": 내용, "type": 바코드/QR코드}}``
      * 실패:``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.detect.detect_qr(self.get_image())
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Detect text
  def search_text(self):
    """
    카메라 이미지 안의 문자를 인식합니다.

    example::

      pibo_edu_v1.search_text()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 인식된 문자열}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.detect.detect_text(self.get_image())
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Detect color
  def search_color(self):
    """
    카메라 이미지(단색 이미지) 안의 색상을 인식합니다.

    (Red, Orange, Yellow, Green, Skyblue, Blue, Purple, Magenta)

    example::

      pibo_edu_v1.search_color()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 인식된 색상}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      img = self.get_image()
      height, width = img.shape[:2]
      img_hls = self.camera.bgr_hls(img)
      cnt, sum_hue = 0, 0
      
      # 평균 패치 측정(j: Height, i: width)
      for i in range(50, width-50, 20):
        for j in range(50, height-50, 20):
          sum_hue += (img_hls[j, i, 0]*2)
          cnt += 1

      hue = round(sum_hue/cnt)

      if ( 0 <= hue <= 30) or (330 <=  hue <= 360):
        ans = "Red"
      elif (31 <=  hue <= 59):
        ans = "Orange"
      elif (60 <=  hue <= 85):
        ans = "Yellow"
      elif (86 <=  hue <= 159):
        ans = "Green"
      elif (160 <=  hue <= 209):
        ans = "Skyblue"
      elif (210 <=  hue <= 270):
        ans = "Blue"
      elif (271 <=  hue <= 290):
        ans = "Purple"
      elif (291<=  hue <= 329):
        ans = "Magenta"
      return self.return_msg(True, "Success", "Success", ans)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Detect face
  def detect_face(self):
    """
    카메라 이미지 안의 얼굴을 탐색합니다.

    example::

      pibo_edu_v1.detect_face()

    :returns:

      * 성공:``{"result": True, "errcode": 0, "errmsg": "Success", "data": 얼굴 좌표(startX, startY, endX, endY)}``
      * 실패:``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      result = self.face.detect(self.get_image())
      return self.return_msg(True, "Success", "Success", result)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Recognize face
  def search_face(self, filename="face.png"):
    """
    카메라 이미지 안의 얼굴을 인식하여 성별과 나이를 추측하고, facedb를 바탕으로 인식한 얼굴의 이름과 정확도를 제공합니다.

    얼굴 인식에 성공하면, 사진을 캡쳐 후 얼굴 위치와 이름, 나이, 성별을 기입 후 `filename` 에 저장합니다.

    (인식한 얼굴 중 가장 크게 인식한 얼굴에 적용됩니다.)

    example::

      pibo_edu_v1.search_face("/home/pi/test.png")

    :param str filename: 저장할 파일 경로

      (이미지 파일 형식 기입 필수 - jpg, jpeg, png, bmp)

    :returns:

      * 성공:
        {"result": True, "errcode": 0, "errmsg": "Success", 
        "data": {"name": 이름, "score": 정확도, "gender": 성별, "age": 나이}}
        # 정확도 0.4 이하 동일인 판정

      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if filename.split('.')[-1] not in ("png", "jpg", "jpeg", "bmp"):
        return self.return_msg(False, "Argument error", f"{filename} must be (png|jpg|jpeg|bmp)", None)

      max_w = -1
      selected_face = []

      img = self.get_image()
      faceList = self.face.detect(img)

      if len(faceList) < 1:
        return self.return_msg(True, "Success", "Success", "No Face")
      for i, (x,y,w,h) in enumerate(faceList):
        if w > max_w:
          max_w = w
          idx = i

      ret = self.face.get_ageGender(img, faceList[idx])
      age = ret["age"]
      gender = ret["gender"]

      x,y,w,h = faceList[idx]
      self.camera.rectangle(img, (x, y), (x+w, y+h))

      ret = self.face.recognize(img, faceList[idx])
      name = "Guest" if ret == False else ret["name"]
      score = "-" if ret == False else ret["score"]
      result = self.camera.putText(img, "{} / {} {}".format(name, gender, age), (x-10, y-10), size=0.5)
      self.camera.imwrite(filename, result)
      return self.return_msg(True, "Success", "Success", {"name": name, "score": score, "gender": gender, "age": age})
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Train face
  def train_face(self, name=None):
    """
    사진 촬영 후 얼굴을 학습합니다. (인식된 얼굴 중 가장 크게 인식한 얼굴에 적용됩니다.)

    example::

      pibo_edu_v1.train_face("kim")

    :param str name: 학습할 얼굴의 이름

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

    """

    try:
      if name == None:
        return self.return_msg(False, "Argument error", "Name is required", None)

      max_w = -1
      img = self.get_image()
      faces = self.face.detect(img)

      if len(faces) < 1:
        return self.return_msg(True, "Success", "Success", "No Face")

      for i, (x,y,w,h) in enumerate(faces):
        if w > max_w:
          max_w = w
          idx = i
      self.face.train_face(img, faces[idx], name)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Delete face in the facedb
  def delete_face(self, name=None):
    """
    facedb에 등록된 얼굴을 삭제합니다.

    example::

      pibo_edu_v1.delete_face("kim")

    :param str name: 삭제할 얼굴 이름

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if name == None:
        return self.return_msg(False, "Argument error", "Name is required", None)

      ret = self.face.delete_face(name)
      if ret == False:
        return self.return_msg(ret, "Runtime error", f"{name} not exist in the facedb", None)
      return self.return_msg(ret, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Get facedb
  def get_facedb(self):
    """
    사용 중인 facedb를 확인합니다.

    example::

      pibo_edu_v1.get_facedb()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 현재 사용 중인 facedb}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      facedb = self.face.get_db()
      return self.return_msg(True, "Success", "Success", facedb)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Save the facedb as a file
  def save_facedb(self, filename=None):
    """
    facedb를 파일로 저장합니다.

    example::

      pibo_edu_v1.save_facedb("/home/pi/facedb")

    :param str filename: 저장할 데이터베이스 파일 경로

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if filename == None:
        return self.return_msg(False, "Argument error", "filename is required", None)

      self.face.save_db(filename)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Reset facedb
  def init_facedb(self):
    """
    facedb를 초기화합니다.

    example::

      pibo_edu_v1.init_facedb()

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      self.face.init_db()
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Load facedb
  def load_facedb(self, filename=None):
    """
    facedb를 불러옵니다.

    example::

      pibo_edu_v1.load_facedb("/home/pi/facedb")

    :param str filename: 불러올 데이터베이스 파일 경로

    :returns:

      * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
      * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
    """

    try:
      if filename == None:
        return self.return_msg(False, "Argument error", "filename is required", None)
      if Path(filename).is_file() == False:
        return self.return_msg(False, "NotFound error", f"{filename} does not exist", None)

      self.face.load_db(filename)
      return self.return_msg(True, "Success", "Success", None)
    except Exception as e:
      return self.return_msg(False, "Exception error", e, None)


  # [Vision] - Determine image
  def get_image(self):
    """
    (내부함수 입니다.)

    카메라로부터 현재 이미지를 가져옵니다.
    """

    return self.img if self.camera_loop else self.camera.read() 


  # Return msg form
  def return_msg(self, status, errcode, errmsg, data):
    """
    (내부함수 입니다.)

    정규 return 메시지 양식을 만듭니다.
    """

    return {"result": status, "errcode": Pibo.return_msg_list[errcode], "errmsg": errmsg, "data": data}

