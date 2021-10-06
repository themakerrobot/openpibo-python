"""
각 메소드의 반환값은 다음과 같은 형식으로 구성됩니다.

* 실행 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": data 또는 None}``

    * 메소드에서 반환되는 데이터가 있을 경우 해당 데이터가 출력되고, 없으면 None이 출력됩니다.

* 실행 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``

    * ``errcode`` 에 err 숫자 코드가, ``errmsg`` 에 해당 error 발생 원인이 출력됩니다.
    * err 숫자코드의 의미와 발생 원인은 다음과 같습니다.

        *  ``0`` : 메소드 실행 성공
        * ``-1`` : Argument error - 메소드 실행에 필요한 필수 인자 값 미기입
        * ``-2`` : Extension error - filename에 확장자 미기입 또는 잘못된 확장자 형식 입력
        * ``-3`` : NotFound error - 존재하지 않는 데이터 입력
        * ``-4`` : Exist error - 이미 존재하는 데이터의 중복 생성
        * ``-5`` : Range error - 지정된 범위를 벗어난 값 입력
        * ``-6`` : Running error - 이미 실행 중인 함수의 중복 사용
        * ``-7`` : Syntax error - 잘못된 형식의 인자 값 입력
        * ``-8`` : Exception error - 위 error 이외의 다른 이유로 메소드 실행에 실패한 경우
"""

import sys, time, pickle

from .audio import Audio
from .oled import Oled
from .speech import Speech, Dialog
from .device import Device
from .motion import Motion
from .vision import Camera, Face, Detect
from .modules.vision.stream import VideoStream

from threading import Thread
from queue import Queue
from pathlib import Path


class Pibo:
    """
    ``openpibo`` 의 다양한 기능들을 한번에 사용할 수 있는 클래스 입니다.

    다음 클래스의 기능을 모두 사용할 수 있습니다.

    * Device
    * Audio
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
    code_list = {
        "Success": 0,
        "Argument error": -1,
        "Extension error": -2,
        "NotFound error": -3,
        "Exist error": -4,
        "Range error": -5,
        "Running error": -6,
        "Syntax error": -7,
        "Exception error": -8,
    }
    """
    반환되는 ``errmsg`` 에 대한 ``errcode`` 입니다.
    """

    def __init__(self):
        self.onair = False
        self.img = ""
        self.check = False
        self.flash = False
        self.device = Device()
        self.audio = Audio()
        self.oled = Oled()
        self.speech = Speech()
        self.dialog = Dialog()
        self.motion = Motion()
        self.camera = Camera()
        self.face = Face()
        self.detect = Detect()
        self.que = Queue()
        self.colordb = {
            'black': (0,0,0),
            'white': (255,255,255),
            'red': (255,0,0),
            'orange': (200,75,0),
            'yellow': (255,255,0),
            'green': (0,255,0),
            'blue': (0,0,255),
            'aqua': (0,255,255),
            'purple': (255,0,255),    
            'pink': (255,51,153),
        }
        self.motor_range = [25,35,80,30,50,25,25,35,80,30]
        self.device.send_cmd(self.device.code['PIR'], "on")

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

        if filename != None:
            file_list = ('mp3', 'wav')
            ext = filename.rfind('.')
            file_ext = filename[ext+1:]
            if file_ext not in file_list:
                return self.return_msg(False, "Extension error", "Audio filename must be 'mp3', 'wav'", None)
            file_exist = self.check_file(filename)
            if file_exist == False:
                return self.return_msg(False, "NotFound error", "The filename does not exist", None)
        else:
            return self.return_msg(False, "Argument error", "Filename is required", None)   
        try:
            if out not in ("local", "hdmi", "both"):
                return self.return_msg(False, "NotFound error", "Output device must be 'local', 'hdmi', 'both'", None)
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


    # [Neopixel] - Determine number or letter    
    def isAlpha(self, *value):
        """
        (``eye_on`` 의 내부함수)

        ``eye_on`` 메소드에서 입력값이 숫자인지 문자인지 판단하기 위한 메소드 입니다.
        """
        
        alpha_cnt = 0

        if len(value) == 1 and type(*value) == str:
            return True
        else:
            for i in value:
                if str(i).isalpha():
                    alpha_cnt += 1
            if alpha_cnt == 0:
                return False
            return True


    # [Neopixel] - LED ON
    def eye_on(self, *color):
        """
        LED를 켭니다.

        example::

            pibo_edu_v1.eye_on(255,0,0)	# 양쪽 눈 제어
            pibo_edu_v1.eye_on(0,255,0,0,0,255) # 양쪽 눈 개별 제어
            pibo_edu_v1.eye_on('Red') # 양쪽 눈 제어('RED', 'red' 가능)
            pibo_edu_v1.eye_on('aqua', 'pink') # 양쪽 눈 개별 제어
        
        :param color:
        
            * RGB (0~255 숫자)
            * color name (영어 대소문자 모두 가능)

                color_list::
                    
                    black, white, red, orange, yellow, green, blue, aqua, purple, pink
        
            두 가지 방식을 동시에 사용할 수 없습니다::

                pibo.eye_on('blue', 0, 255, 0) (X)
        
        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if len(color) == 0:
            return self.return_msg(False, "Argument error", "RGB or Color is required", None)
        try:
            if self.isAlpha(*color) == False:
                for i in color:
                    if i < 0 or i > 255:
                        return self.return_msg(False, "Range error", "RGB value should be 0~255", None)
                if len(color) == 3:
                    cmd = "#20:{}!".format(",".join(str(p) for p in color))
                elif len(color) == 6:
                    cmd = "#23:{}!".format(",".join(str(p) for p in color))
                else:
                    return self.return_msg(False, "Syntax error", "Only 3 or 6 values can be entered", None)
            else:
                if len(color) == 1:
                    color = color[-1].lower()
                    if color not in self.colordb.keys():
                        return self.return_msg(False, "NotFound error", "{} does not exist in the colordb".format(color), None)
                    else:
                        color = self.colordb[color]
                        cmd = "#20:{}!".format(",".join(str(p) for p in color))
                elif len(color) == 2:
                    l_color, r_color = color[0].lower(), color[1].lower()
                    if l_color in self.colordb.keys() and r_color in self.colordb.keys():
                        l_color = self.colordb[l_color]
                        r_color = self.colordb[r_color]
                        color = l_color + r_color
                        cmd = "#23:{}!".format(",".join(str(p) for p in color))
                    else:
                        if l_color not in self.colordb.keys():
                            return self.return_msg(False, "NotFound error", "{} does not exist in the colordb".format(color[0]), None)
                        return self.return_msg(False, "NotFound error", "{} does not exist in the colordb".format(color[1]), None)
                else:
                    return self.return_msg(False, "Syntax error", "Only 2 colors can be entered", None)
            if self.check:
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
            cmd = "#20:0,0,0!"
            if self.check:
                self.que.put(cmd)
            else:
                self.device.send_raw(cmd)
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Create the color
    def add_color(self, color=None, *rgb):
        """
        ``colordb`` 에 원하는 색상을 추가합니다.

        example::

            pibo_edu_v1.add_color('sky', 85, 170, 255)

        :param str color: 추가할 색상 이름

        :param rgb: RGB. 0~255 사이의 정수 입니다.

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if color == None or type(color) != str:
            return self.return_msg(False, "Argument error", "Color is required", None)
        else:
            if not rgb:
                return self.return_msg(False, "Argument error", "RGB value is required", None)
            else:
                if len(rgb) != 3:
                    return self.return_msg(False, "Syntax error", "3 values are required(R,G,B)", None)
                for i in rgb:
                    if i < 0  or i > 255:
                        return self.return_msg(False, "Range error", "RGB value should be 0~255", None)
        try:
            color_list = self.get_colordb()["data"]
            if color in color_list.keys():
                return self.return_msg(False, "Exist error", "{} is already exist".format(color), None)
            color_list[color] = rgb
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Get colordb
    def get_colordb(self):
        """
        사용 중인 ``colordb`` 를 확인합니다.
        
        (``pibo.eye_on()`` 에 입력할 수 있는 color 목록 조회)

        example::

            pibo_edu_v1.get_colordb()

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": 현재 사용 중인 colordb}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            return self.return_msg(True, "Success", "Success", self.colordb)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Reset colordb
    def init_colordb(self):
        """
        ``colordb`` 를 기존에 제공하는 ``colordb`` 상태로 초기화합니다.

        example::

            pibo_edu_v1.init_colordb()

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            self.colordb = {
                'black': (0,0,0),
                'white': (255,255,255),
                'red': (255,0,0),
                'orange': (200,75,0),
                'yellow': (255,255,0),
                'green': (0,255,0),
                'blue': (0,0,255),
                'aqua': (0,255,255),
                'purple': (255,0,255),    
                'pink': (255,51,153),
            }
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Save the colordb as a file
    def save_colordb(self, filename=None):
        """
        ``colordb`` 를 파일로 저장합니다.

        example::

            pibo_edu_v1.save_colordb('/home/pi/new_colordb')

        :param str filename: 저장할 데이터베이스 파일 경로

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if filename == None:
            return self.return_msg(False, "Argument error", "Filename is required", None)
        try:
            with open(filename, "w+b") as f:
                pickle.dump(self.colordb, f)
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Load colordb
    def load_colordb(self, filename=None):
        """
        ``colordb`` 를 불러옵니다.

        example::

            pibo_edu_v1.load_colordb('/home/pi/new_colordb')
        
        :param str filename: 불러올 데이터베이스 파일 경로

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if filename == None:
            return self.return_msg(False, "Argument error", "Filename is required", None)
        else:
            file_exist = self.check_file(filename)
            if file_exist == False:
                return self.return_msg(False, "NotFound error", "The filename does not exist", None)
        try:
            with open(filename, "rb") as f:
                self.colordb = pickle.load(f)
                return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Neopixel] - Delete color in the colordb
    def delete_color(self, color=None):
        """
        ``colordb`` 에 등록된 색상을 삭제합니다.

        example::

            pibo_edu_v1.delete_color('red')
        
        :param str color: 삭제할 색상 이름

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if color == None:
            return self.return_msg(False, "Argument error", "Color is required", None)
        try:
            ret = color in self.colordb.keys()
            if ret == True:
                del self.colordb[color]
                return self.return_msg(ret, "Success", "Success", None)
            return self.return_msg(False, "NotFound error", "{} not exist in the colordb".format(color), None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Device] - Check device
    def check_device(self, system=None):
        """
        디바이스의 상태를 확인합니다. (일회성)

        example::

            pibo_edu_v1.check_device('battery')

        :param str system: ``system`` / ``battery`` (대문자도 가능)

            ``system`` 입력 시 PIR, TOUCH, DC_CONN, BUTTON의 상태를 조회할 수 있습니다.
        
        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": Device로부터 응답}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        device_list = ('BATTERY', 'SYSTEM')
        if system:
            system = system.upper()
            if system not in device_list:
                return self.return_msg(False, "NotFound error", "System must be 'battery', 'system'", None)
        else:
            return self.return_msg(False, "Argument error", "Enter the device name to check", None)
        try:
            ret = self.device.send_cmd(self.device.code[system])
            idx = ret.find(':')
            if system == "BATTERY":
                ans = system + ret[idx:]
            elif system == "SYSTEM":
                result = ret[idx+1:].split('-')
                ans = {"PIR": "", "TOUCH": "", "DC_CONN": "", "BUTTON": "",}
                ans["PIR"] = result[0]
                ans["TOUCH"] = result[1]
                ans["DC_CONN"] = result[2]
                ans["BUTTON"] = result[3]
            return self.return_msg(True, "Success", "Success", ans)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Device] - start_devices thread
    def thread_device(self, func):
        """
        (``start_devices`` 의 내부함수)

        ``start_devices`` 메소드에서 디바이스의 상태를 파악하여 해당 메시지를 ``func`` 하는 메소드입니다.

        ``func`` 에는 print 함수가 들어가서 메시지가 계속해서 출력(print) 됩니다.
        """

        self.system_check_time = time.time()
        self.battery_check_time = time.time()
        while True:
            if self.check == False:
                break

            if self.que.qsize():
                self.device.send_raw(self.que.get())

            if time.time() - self.system_check_time > 1:
                ret = self.device.send_cmd(self.device.code["SYSTEM"])
                idx = ret.find(':')
                msg = "SYSTEM" + ret[idx:]
                func(msg)
                self.system_check_time = time.time()

            if time.time() - self.battery_check_time > 10:
                ret = self.device.send_cmd(self.device.code["BATTERY"])
                msg = "BATTERY" + ret[idx:]
                func(msg)
                self.battery_check_time = time.time()
            time.sleep(0.01)


    # [Device] - Check device(thread)
    def start_devices(self, func=None):
        """
        디바이스의 상태를 확인합니다.

        example::

            def msg_device(msg):
                print(msg)
            
            def check_devices():
                pibo_edu_v1.start_devices(msg_device)
            
            check_devices()

        :param func: Device로부터 받은 응답을 확인하기 위한 함수

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if func == None:
            return self.return_msg(False, "Argument error", "Func is required", None)
        if self.check:
            return self.return_msg(False, "Running error", "start_devices() is already running", None)
        try:
            self.check = True
            t = Thread(target=self.thread_device, args=(func,))
            t.daemon = True
            t.start()
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Device] - Stop check device
    def stop_devices(self):
        """
        디바이스의 상태 확인을 종료합니다.

        example::

            pibo_edu_v1.stop_devices()

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            self.check = False
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

        if n != None:
            if n < 0 or n > 9:
                return self.return_msg(False, "Range error", "Channel value should be 0~9", None)
        else:
            return self.return_msg(False, "Argument error", "Channel is required", None)
        if position != None:
            if abs(position) > self.motor_range[n]:
                return self.return_msg(False, "Range error", "The position range of channel {} is -{} ~ {}".format(n, self.motor_range[n], self.motor_range[n]), None)
        else:
            return self.return_msg(False, "Argument error", "Position is required", None)
        try:
            if speed != None:
                if speed < 0 or speed > 255:
                    return self.return_msg(False, "Range error", "Speed value should be 0~255", None)
                self.motion.set_speed(n, speed)
            if accel != None:
                if accel < 0 or accel > 255:
                    return self.return_msg(False, "Range error", "Acceleration value should be 0~255", None)
                self.motion.set_acceleration(n, accel)
            self.motion.set_motor(n, position)
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Motion] - Control all motors(position/speed/accel)
    def motors(self, positions=None, speed=None, accel=None):
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

        check = self.check_motor("position", positions)
        if check["result"] == False:
            return check
        try:
            if speed != None:
                check = self.check_motor("speed", speed)
                if check["result"] == False:
                    return check
                self.motion.set_speeds(speed)
            if accel != None:
                check = self.check_motor("acceleration", accel)
                if check["result"] == False:
                    return check
                self.motion.set_accelerations(accel)
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

        check = self.check_motor("position", positions)
        if check["result"] == False:
            return check
        if movetime and movetime < 0:
            return self.return_msg(False, "Range error", "Movetime is only available positive number", None)
        try:
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
    def set_motion(self, name=None, cycle=1):
        """
        모션의 동작을 실행합니다.

        example::

            pibo_edu_v1.set_motion("dance1", 5)

        :param str name: 모션 이름

        :param int cycle: 모션 반복 횟수

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """
        
        if name == None:
            return self.return_msg(False, "Argument error", "Name is required", None)
        try:
            ret = self.motion.set_motion(name, cycle)
            if ret ==  False:
                return self.return_msg(False, "NotFound error", "{} not exist in the motor profile".format(name), None)
            return self.return_msg(ret, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Motion] - Check motors array
    def check_motor(self, mode, values):
        """
        (``motors`` , ``motors_movetime`` 메소드의 내부함수 입니다.)

        모터의 각도를 설정할 때, 허용 각도 범위 내에 해당하는지 판별하기 위한 함수입니다.

        각 모터의 허용 각도 범위는 아래와 같습니다::

            [25,35,80,30,50,25,25,35,80,30]
            # 0번 모터는 -25 ~ 25 범위의 모터 각도를 가집니다.
        """

        try:
            if values == None or len(values) != 10:
                return self.return_msg(False, "Syntax error", "10 {}s are required".format(mode), None)
            if mode == "position":
                for i in range(len(values)):
                    if abs(values[i]) > self.motor_range[i]:
                        return self.return_msg(False, "Range error", "The position range of channel {} is -{} ~ {}".format(i, self.motor_range[i], self.motor_range[i]), None)
            else:
                for v in values:
                    if v < 0 or v > 255:
                        return self.return_msg(False, "Range error", "{} value should be 0~255".format(mode.capitalize()), None)
            return self.return_msg(True, "Success", "Success", None)
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

        check = self.points_check("text", points)
        if check["result"] == False:
            return check
        if text == None or type(text) != str:
            return self.return_msg(False, "Argument error", "Text is required", None)
        try:
            if size:
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
        
        if filename == None:
            return self.return_msg(False, "Argument error", "Filename is required", None)
        else:
            ext = filename.rfind('.')
            file_ext = filename[ext+1:]
            if file_ext != 'png':
                return self.return_msg(False, "Extension error", "Only png files are available", None)
            file_exist = self.check_file(filename)
            if file_exist:
                img_check = self.oled.size_check(filename)
                if img_check[0] != 64 or img_check[1] != 128:
                    return self.return_msg(False, "Syntax error", "Only 128X64 sized files are available", None)
            else:
                return self.return_msg(False, "NotFound error", "The filename does not exist", None)
        try:
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

        :param tuple(int, int, int, int) points: 선 - 시작 좌표, 끝 좌표(x, y, x1, y1)
        
            사각형, 원 - 좌측상단, 우측하단 좌표 튜플(x, y, x1, y1)

        :param str shape: 도형 종류 - ``rectangle`` / ``circle`` / ``line``

        :param bool fill: ``True`` (채움) / ``False`` (채우지 않음)

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """
        
        check = self.points_check("figure", points)
        if check["result"] == False:
            return check
        if shape == None or type(shape) != str:
            return self.return_msg(False, "Argument error", "Shape is required", None)
        try:
            if shape == 'rectangle':
                self.oled.draw_rectangle(points, fill)
            elif shape == 'circle':
                self.oled.draw_ellipse(points, fill)
            elif shape == 'line':
                self.oled.draw_line(points)
            else:
                return self.return_msg(False, "NotFound error", "The shape does not exist", None)
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


    # [OLED] - Check points
    def points_check(self, mode, points=None):
        """
        (``draw_text`` , ``draw_figure`` 의 내부함수 입니다.)

        text와 figure의 위치를 결정하는 points가 적절한 포맷인지 확인하기 위한 메소드입니다.

        text는 point가 2개 (x, y) 필요합니다. (시작지점)

        figure는 point가 4개 (x, y, x1, y1) 필요합니다. (시작지점, 끝지점)
        """

        number = 2
        if mode == "figure":
            number = 4
        try: 
            if points == None or type(points) != tuple:
                return self.return_msg(False, "Argument error", "{} points are required".format(number), None)
            else:
                if len(points) != number:
                    return self.return_msg(False, "Syntax error", "{} points are required".format(number), None)
                for i in points:
                    if i < 0:
                        return self.return_msg(False, "Range error", "Points are only available positive number", None)
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

        to_list = ('ko', 'en')
        if string == None:
            return self.return_msg(False, "Argument error", "String is required", None)
        if to not in to_list:
            return self.return_msg(False, "Syntax error", "Translation is only available 'ko', 'en'", None)
        try:
            ret = self.speech.translate(string, to)
            return self.return_msg(True, "Success", "Success", ret)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Speech] - TTS
    def tts(self, string=None, filename='tts.mp3'):
        """
        Text(문자)를 Speech(음성)로 변환합니다.

        example::

            pibo_edu_v1.tts(
                "<speak><voice name='MAN_READ_CALM'>안녕하세요. 반갑습니다.<break time='500ms'/></voice></speak>", 
                "/home/pi/tts.mp3"
                )

        :param str string: 변환할 문장

            * <speak>

                * 기본적으로 모든 음성은 태그로 감싸져야 합니다.
                * 문장, 문단 단위로 적용하는 것을 원칙으로 합니다.
                
                  한 문장 안에서 단어별로 태그를 감싸지 않습니다.

                * <speak> 안녕하세요. 반가워요. </speak>

            * <voice>

                * 음성의 목소리를 변경하기 위해 사용하며,
                
                  name attribute를 통해 원하는 목소리를 지정합니다. 
                
                  제공되는 목소리는 4가지입니다.

                  * WOMAN_READ_CALM: 여성 차분한 낭독체 (default)
                  * MAN_READ_CALM: 남성 차분한 낭독체
                  * WOMAN_DIALOG_BRIGHT: 여성 밝은 대화체
                  * MAN_DIALOG_BRIGHT: 남성 밝은 대화체

                * 문장, 문단 단위로 적용하는 것을 원칙으로 합니다. 
                
                  한 문장 안에서 단어별로 태그를 감싸지 않습니다.

                example::

                    <speak>
                        <voice name="WOMAN_READ_CALM"> 여성 차분한 낭독체입니다.</voice>
                        <voice name="MAN_READ_CALM"> 남성 차분한 낭독체입니다.</voice>
                        <voice name="WOMAN_DIALOG_BRIGHT"> 여성 밝은 대화체예요.</voice>
                        <voice name="MAN_DIALOG_BRIGHT"> 남성 밝은 대화체예요.</voice>
                    </speak>

        :param str filename: 저장할 파일 경로(mp3, wav)

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        if string == None:
            return self.return_msg(False, "Argument error", "String is required", None)
        ext = filename.rfind('.')
        file_ext = filename[ext+1:]
        if file_ext != 'mp3':
            return self.return_msg(False, "Extension error", "TTS filename must be 'mp3'", None)
        voice_list= ('WOMAN_READ_CALM', 'MAN_READ_CALM', 'WOMAN_DIALOG_BRIGHT', 'MAN_DIALOG_BRIGHT')
        if '<speak>' not in string or '</speak>' not in string:
            return self.return_msg(False, "Syntax error", "Invalid string format", None)
        elif '<voice' in string and '</voice>' in string:
            voice_start = string.find('=')
            voice_end = string.find('>', voice_start)
            voice_name = string[voice_start+2:voice_end-1]
            if voice_name not in voice_list:
                return self.return_msg(False, "NotFound error", "The voice name does not exist", None)
        try:
            ret = self.speech.tts(string, filename)
            if ret == False:
                return self.return_msg(False, "Exception error", "openpibo.speech.tts function error", None)
            return self.return_msg(True, "Success", "Success", ret)
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
            ret = self.speech.stt(filename, timeout)
            if ret == False:
                return self.return_msg(False, "Exception error", "openpibo.speech.stt function error", None)
            return self.return_msg(True, "Success", "Success", ret)
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

        if q:
            if type(q) != str:
                return self.return_msg(False, "Syntax error", "Q is only available str type", None)
        else:
            return self.return_msg(False, "Argument error", "Q is required", None)
        try:
            ret = self.dialog.get_dialog(q)
            return self.return_msg(True, "Success", "Success", ret)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Vision] - start_camera thread
    def camera_on(self):
        """
        (``start_camera`` 메소드의 내부함수 입니다.)

        카메라로 짧은 주기로 사진을 찍어 128x64 크기로 변환한 후 OLED에 보여줍니다.
        """

        vs = VideoStream().start()

        while True:
            if self.onair == False:
                vs.stop()
                break
            self.img = vs.read()
            img = self.img
            img = self.camera.convert_img(img, 128, 64)
            #_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            if self.flash:
                img = self.camera.rotate10(img)
                self.oled.draw_data(img)
                self.oled.show()
                time.sleep(0.3)
                self.flash = False
                continue
            self.oled.draw_data(img)
            self.oled.show()


    # [Vision] - Camera ON
    def start_camera(self):
        """
        카메라가 촬영하는 영상을 OLED에 보여줍니다.

        example::

            pibo_edu_v1.start_camera()

        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            if self.onair:
                return self.return_msg(False, "Running error", "start_camera() is already running", None)
            self.onair = True
            t = Thread(target=self.camera_on, args=())
            t.daemon = True
            t.start()
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Vision] - Camera OFF
    def stop_camera(self):
        """
        카메라를 종료합니다.

        example::

            pibo_edu_v1.stop_camera()
        
        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": None}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            self.onair = False
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

        file_list = ("png", "jpg", "jpeg", "bmp")
        ext = filename.rfind('.')
        file_ext = filename[ext+1:]
        if file_ext not in file_list:
            return self.return_msg(False, "Extension error", "Image filename must be 'png', 'jpg', 'jpeg', 'bmp'", None) 
        try:
            if self.onair:
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

            * 성공::
            
                {
                    "result": True, "errcode": 0, "errmsg": "Success", 
                    "data": {
                        "name": 이름, "score": 점수, 
                        "position": 사물좌표(startX, startY, endX, endY)
                    }
                }

            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            img = self.check_onair()
            ret = self.detect.detect_object(img)
            return self.return_msg(True, "Success", "Success", ret)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Vision] - Detect QR/barcode
    def search_qr(self):
        """
        카메라 이미지 안의 QR 코드 및 바코드를 인식합니다.

        example::

            pibo_edu_v1.search_qr()

        :returns:
        
            * 성공::
            
                {
                    "result": True, "errcode": 0, "errmsg": "Success", 
                    "data": {"data": 내용, "type": 바코드/QR코드}
                }

            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            img = self.check_onair()
            ret = self.detect.detect_qr(img)
            return self.return_msg(True, "Success", "Success", ret)
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
            img = self.check_onair()
            ret = self.detect.detect_text(img)
            return self.return_msg(True, "Success", "Success", ret)
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
            img = self.check_onair()
            height, width = img.shape[:2]
            img_hls = self.camera.bgr_hls(img)
            cnt = 0
            sum_hue = 0
            
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

            * 성공::
            
                {
                    "result": True, "errcode": 0, "errmsg": "Success", 
                    "data": 얼굴 좌표(startX, startY, endX, endY)
                }

            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        try:
            img = self.check_onair()
            faceList = self.face.detect(img)
            if len(faceList) < 1:
                return self.return_msg(True, "Success", "Success", "No Face")
            return self.return_msg(True, "Success", "Success", faceList)
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

            * 성공::

                {"result": True, "errcode": 0, "errmsg": "Success", 
                "data": {"name": 이름, "score": 정확도, "gender": 성별, "age": 나이}}
                # 정확도 0.4 이하 동일인 판정

            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        max_w = -1
        selected_face = []
        if filename != None:
            file_list = ("png", "jpg", "jpeg", "bmp")
            ext = filename.rfind('.')
            file_ext = filename[ext+1:]
            if file_ext not in file_list:
                return self.return_msg(False, "Extension error", "Image filename must be 'png', 'jpg', 'jpeg', 'bmp'", None)
        try:
            img = self.check_onair()
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

        max_w = -1
        if name == None:
            return self.return_msg(False, "Argument error", "Name is required", None)
        try:
            img = self.check_onair()
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

        if name == None:
            return self.return_msg(False, "Argument error", "Name is required", None)
        try:
            ret = self.face.delete_face(name)
            if ret == False:
                return self.return_msg(ret, "NotFound error", "{} not exist in the facedb".format(name), None)
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

        if filename == None:
            return self.return_msg(False, "Argument error", "Filename is required", None)
        try:
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

        if filename == None:
            return self.return_msg(False, "Argument error", "Filename is required", None)
        else:
            file_exist = self.check_file(filename)
            if file_exist == False:
                return self.return_msg(False, "NotFound error", "The filename does not exist", None)
        try:
            self.face.load_db(filename)
            return self.return_msg(True, "Success", "Success", None)
        except Exception as e:
            return self.return_msg(False, "Exception error", e, None)


    # [Vision] - Determine image
    def check_onair(self):
        """
        (내부함수 입니다.)

        카메라로부터 현재 이미지를 가져옵니다.
        """
        if self.onair:
            img = self.img
        else:
            img = self.camera.read()
        return img


    # Check file exist
    def check_file(self, filename):
        """
        (내부함수 입니다.)

        파일을 불러올 때, 불러오려는 파일이 존재하는지 여부를 판단합니다.
        """

        return Path(filename).is_file()

    
    # Return msg form
    def return_msg(self, status, errcode, errmsg, data):
        """
        (내부함수 입니다.)

        정규 return 메시지 양식을 만듭니다.
        """

        return {"result": status, "errcode": Pibo.code_list[errcode], "errmsg": errmsg, "data": data}


    # Getting the meaning of error code
    def get_codeMean(self, errcode):
        """
        err 숫자코드의 의미를 조회합니다.

        example::

            pibo_edu_v1.get_codeMean(-3)

        :param int errcode: 조회하고자 하는 errcode 숫자
        
        :returns:

            * 성공: ``{"result": True, "errcode": 0, "errmsg": "Success", "data": errcode 의미}``
            * 실패: ``{"result": False, "errcode": errcode, "errmsg": "errmsg", "data": None}``
        """

        n_list = {value:key for key, value in Pibo.code_list.items()}

        if errcode in n_list.keys():
            return self.return_msg(True, "Success", "Success", n_list[errcode])
        return self.return_msg(False, "NotFound error", "Error code {} does not exist".format(errcode), None)
