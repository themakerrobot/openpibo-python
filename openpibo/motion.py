"""
PIBO의 움직임을 제어합니다.

Class:
:obj:`~openpibo.motion.Motion`
:obj:`~openpibo.motion.PyMotion`

:모터 번호당 위치:

  * 0번 : 'Right Foot'
  * 1번 : 'Right Leg'
  * 2번 : 'Right Arm'
  * 3번 : 'Right Hand'
  * 4번 : 'Head Pan'
  * 5번 : 'Head Tilt'
  * 6번 : 'Left Foot'
  * 7번 : 'Left Leg'
  * 8번 : 'Left Arm'
  * 9번 : 'Left Hand'

  **(파이보 기준으로 Right, Left 입니다.)**

:모터 제한 각도: 각 모터가 회전할 수 있는 범위가 제한되어있습니다.

  * 0번 : ± 25˚
  * 1번 : ± 35˚
  * 2번 : ± 80˚
  * 3번 : ± 30˚
  * 4번 : ± 50˚
  * 5번 : ± 25˚
  * 6번 : ± 25˚
  * 7번 : ± 35˚
  * 8번 : ± 80˚
  * 9번 : ± 30˚

:모션 데이터베이스: 모션 데이터가 저장되어있는 JSON형태의 데이터입니다.

  모션 데이터베이스는 다음 형식을 갖추고 있습니다::

    {
      "name": {
        "comment":"description of this motion",
        "init_def":0,
        "init":[0,0,-70,-25,0,0,0,0,70,25],
        "pos":[
          { "d": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] , "seq": 0 }
        ]
      }
    }

:모션 프로파일: 모션 데이터베이스로부터 가져온 모션 데이터를 인스턴스에 저장한 것. ``set_motion`` 메소드로 프로파일 내의 동작을 수행할 수 있습니다.

  모션 프로파일은 각 인스턴스에 저장되며, 인스턴스 초기화 시 인스턴스 변수 ``profile`` 에 기본 모션 데이터베이스가 저장됩니다.

  기본 모션 데이터베이스 내 모션 리스트::

    stop, stop_body, sleep, lookup, left, left_half, right, right_half,
    foward1-2, backward1-2, step1-2, hifive, cheer1-3, wave1-6, think1-4,
    wake_up1-3, hey1-2, yes_h, no_h, breath1-3, breath_long, head_h,
    spin_h, clapping1-2, hankshaking, bow, greeting, hand1-4, foot1-2,
    speak1-2, speak_n1-2, speak_q, speak_r1-2, speak_l1-2, welcome,
    happy1-3, excite1-2, boring1-2, sad1-3, handup_r, handup_l, look_r,
    look_l, dance1-5, motion_test, test1-4
    # foward1-2는 forward1, forward2 두 종류가 있음을 의미합니다.
"""

import serial
import time
import os
import json

import openpibo_models
#current_path = os.path.dirname(os.path.abspath(__file__))

class Motion:
  """
Functions:
:meth:`~openpibo.motion.Motion.set_profile`
:meth:`~openpibo.motion.Motion.set_motor`
:meth:`~openpibo.motion.Motion.set_motors`
:meth:`~openpibo.motion.Motion.set_speed`
:meth:`~openpibo.motion.Motion.set_speeds`
:meth:`~openpibo.motion.Motion.set_acceleration`
:meth:`~openpibo.motion.Motion.set_accelerations`
:meth:`~openpibo.motion.Motion.get_motion`
:meth:`~openpibo.motion.Motion.set_motion_raw`
:meth:`~openpibo.motion.Motion.set_motion`
:meth:`~openpibo.motion.Motion.stop`

  파이보의 움직임을 제어합니다.

  example::

    from openpibo.motion import Motion

    pibo_motion = Motion()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    """Motion 클래스 초기화"""

    self.profile_path=openpibo_models.filepath("motion_db.json")
    #self.profile_path=current_path+"/data/models/motion_db.json"
    with open(self.profile_path, 'r') as f:
      self.profile = json.load(f)

  def set_profile(self, filename):
    """
    만들어둔 모션 데이터베이스를 불러와 모션 프로파일에 저장합니다.

    모션은 **openpibo-tools** 의 `motion_creator <https://themakerrobot.github.io/x-openpibo/build/html/tools/motion_creator.html>`_ 를 이용해 생성할 수 있습니다.

    example::

      # 불러올 모션 데이터베이스의 경로가 /home/pi/mydata/motion.json 라면,

      pibo_motion.set_profile('/home/pi/mydata/motion.json')

    :param str filename: 모터 프로파일 경로
    """

    if not os.path.isfile(filename):
      raise Exception(f'"{filename}" does not exist')

    with open(filename, 'r') as f:
      self.profile = json.load(f)

  def set_motor(self, n, pos):
    """
    모터 1개를 특정 위치로 이동합니다.

    example::

      pibo_motion.set_motor(2, 30)

    :param int n: 모터 번호

      0~9의 숫자가 들어갑니다.
      해당 번호의 위치는 상단의 ``모터 번호당 위치`` 를 참고해주세요.

    :param int pos: 모터 각도

      -80~80의 숫자가 들어갑니다.
      자세한 범위는 상단의 ``모터 제한 각도`` 를 참고해주세요.
    """

    if type(n) is not int:
      raise Exception (f'"{n}" must be integer type')

    if abs(n) > 9:
      raise Exception (f'"{n}" must be 0~9')

    if type(pos) is not int:
      raise Exception(f'"{pos}" must be integer type')

    os.system(f"servo write {n} {pos*10}")

  def set_motors(self, positions, movetime=None):
    """
    전체 모터를 특정 위치로 이동합니다.

    movetime이 짧을수록 모션을 취하는 속도가 빨라집니다.
    만약 ``movetime`` 이 ``None`` 이라면, 속도는 이전 설정값으로 유지됩니다.

    example::

      pibo_motion.set_motors([0, 0, -80, 0, 0, 0, 0, 0, 80, 0])

    :param list positions: 0-9번 모터 각도 배열

    :param int movetime: 모터 이동 시간(ms)

      50ms 단위, 모터가 정해진 위치까지 이동하는 시간

      (모터 컨트롤러와의 overhead문제로 정밀하지는 않음)
    """

    if len(positions) != 10:
      raise Exception (f'len({positions}) must be 10')

    mpos = [positions[i]*10 for i in range(len(positions))]

    if movetime == None:
      os.system(f'servo mwrite {" ".join(map(str, mpos))}')
    else:
      os.system(f'servo move {" ".join(map(str, mpos))} {movetime}')

  def set_speed(self, n, speed):
    """
    모터 1개의 속도를 설정합니다.

    example::

      pibo_motion.set_speed(3, 255)

    :param int n: 모터 번호

    :param int speed: 모터 속도

      0~255 사이 값입니다.
      숫자가 클수록 속도가 빨라집니다.
    """

    if type(n) is not int:
      raise Exception (f'"{n}" must be integer type')

    if abs(n) > 9:
      raise Exception (f'"{n}" must be 0~9')

    if type(speed) is not int:
      raise Exception (f'"{speed}" must be integer type')

    if abs(speed) > 255:
      raise Exception (f'"{speed}" must be 0~255')

    os.system(f'servo speed {n} {speed}')

  def set_speeds(self, speeds):
    """
    전체 모터의 속도를 설정합니다.

    example::

      pibo_motion.set_speeds([20, 50, 40, 20, 20, 10, 20, 50, 40, 20])

    :param list speeds: 0-9번 모터 속도 배열

      배열 안의 각 가속도는 0~255 사이 정수입니다.
    """

    if len(speeds) != 10:
      raise Exception (f'len({speeds}) must be 10')

    os.system(f'servo speed all {" ".join(map(str, speeds))}')

  def set_acceleration(self, n, accel):
    """
    모터 1개의 가속도를 설정합니다.

    가속도를 설정하게 되면, 속도가 0에서 시작하여 설정된 속도까지 점점 빨라집니다.
    그리고 점점 느려지다가 종료 지점에서 속도가 0이 됩니다.

    example::

      pibo_motion.set_acceleration(3, 5)

    :param int n: 모터 번호

    :param int accel: 모터 속도

      0~255 사이 값입니다.
      숫자가 클수록 가속도가 커집니다.
    """

    if type(n) is not int:
      raise Exception (f'"{n}" must be integer type')

    if abs(n) > 9:
      raise Exception (f'"{n}" must be 0~9')

    if type(accel) is not int:
      raise Exception (f'"{accel}" must be integer type')

    if abs(accel) > 255:
      raise Exception (f'"{accel}" must be 0~255')

    os.system(f'servo accelerate {n} {accel}')

  def set_accelerations(self, accels):
    """
    전체 모터의 가속도를 설정합니다.

    example::

      pibo_motion.set_accelerations([5, 5, 5, 5, 10, 10, 5, 5, 5, 5])

    :param list accels: 0-9번 모터 가속도 배열

      배열 안의 각 가속도는 0~255 사이 정수입니다.
    """

    if len(accels) != 10:
      raise Exception (f'len({accels}) must be 10')

    os.system(f'servo accelerate all {" ".join(map(str, accels))}')

  def get_motion(self, name=None):
    """
    모션 프로파일을 조회합니다.

    ``name`` 매개변수가 ``None`` 이면, 현재 모션 프로파일에 저장되어있는 모든 moiton 이름을 출력하고,
    ``None`` 이 아니면, 해당 이름의 모션에 대한 데이터를 출력합니다.

    example::

      pibo_motion.get_motion('forward1')

    :param str name: 동작 이름

      profile에 저장되어있는 동작의 이름입니다.

      초기화 시 forward1(전진1), forward2(전진2), sleep(잠자기), dance1(춤추기1) 등을 조회할 수 있습니다.

    :returns:

      * ``name == None`` 인 경우::

          # pibo_motion.get_motion()

          ['stop', 'stop_body', 'sleep', 'lookup', 'left', 'left_half',
          'right', 'right_half', 'forward1', 'forward2', ...]

      * ``name != None`` 인 경우::

          # pibo_motion.get_motion('stop')

          {
            'comment': 'stop',
            'init_def': 1,
            'init': [0, 0, -70, -25, 0, 0, 0, 0, 70, 25]
          }
    """

    return list(self.profile.keys()) if name == None else self.profile.get(name)

  def set_motion_raw(self, exe, cycle=1):
    """
    모션 프로파일의 동작을 실행합니다.

    example::

      pibo_motion.set_motion_raw(
        {'init_def': 1, 'init': [0, 0, -70, -25, 0, 0, 0, 0, 70, 25]},
        1
      )

    :param str exe: 특정 동작을 위한 각 모터의 움직임이 기록된 데이터 입니다.

      다음과 같은 양식을 따릅니다::

        {
          "init_def": 1
          "init":[0,0,-70,-25,0,0,0,0,70,25],
          "pos":[
            {"d":[999,999,999,999, 30,999,999,999,999,999],"seq":500},
            {"d":[999,999,999,999,-30,999,999,999,999,999],"seq":2000},
            ...
          ]
        }

      * ``init_def`` 는 초기동작의 유무입니다.
      * ``init`` 은 동작을 시작하기 전의 준비동작입니다.
      * ``pos`` 는 연속된 동작이 담긴 list 입니다.
        **seq** 시간(ms)에 **d** 의 동작이 완료됨을 의미합니다.

    :param int cycle:

      동작을 몇 번 반복할지 결정합니다.
    """

    if exe == None:
      raise Exception(f'"{exe}" is not motion data')

    if type(cycle) is not int or cycle < 1:
      raise Exception(f'"{cycle} must be integer type and positive number')

    seq,cnt,cycle_cnt = 0,0,0
    self.stopped = False

    if exe["init_def"] == 1:
      self.set_motors(exe["init"], seq)

    if "pos" not in exe:
      return

    time.sleep(0.5)
    while True:
      if self.stopped:
        break

      d, intv = exe["pos"][cnt]["d"], exe["pos"][cnt]["seq"]-seq
      seq = exe["pos"][cnt]["seq"]

      self.set_motors(d, intv)
      time.sleep(intv/1000)
      cnt += 1
      if cnt == len(exe["pos"]):
        cycle_cnt += 1
        if cycle > cycle_cnt:
          cnt,seq = 0,0
          continue
        break

  def set_motion(self, name, cycle=1, profile_path=None):
    """
    모션 프로파일의 동작을 실행하는 ``set_motion_raw`` 메소드를 실행합니다.
    ``motion_db`` 에서 ``name`` 에 해당하는 **JSON** 형식의 데이터를 불러와 ``set_motion_raw`` 메소드에게 넘겨줍니다.

    example::

      pibo_motion.set_motion('dance1')

    :param str name: 동작 이름

      모션 프로파일에 저장되어있는 동작의 이름입니다.

      초기화 시 기본 모션 프로파일에 있는 동작(forward1, forward2, sleep, dance1 등)을 실행할 수 있습니다.

    :param int cycle:

      동작 반복 횟수

    :param str profile_path:

      커스텀 동작 프로파일 경로입니다. 입력하지 않으면 기본 프로파일을 불러옵니다.
    """

    if profile_path == None:
      result = self.profile.get(name)
    elif os.path.isfile(profile_path):
      with open(profile_path, 'r') as f:
        result = json.load(f).get(name)
    else:
      raise Exception(f'"{profile_path}" does not exist')

    if result == None:
      raise Exception(f'"{name}" does not exist in motion profile')
    return self.set_motion_raw(result, cycle)

  def stop(self):
    """
    수행 중인 동작을 정지합니다.

    example::

      # 동작을 수행 중일 때,
      pibo_motion.stop()
    """
    self.stopped = True


class PyMotion:
  """
Functions:
:meth:`~openpibo.motion.PyMotion.set_motor`
:meth:`~openpibo.motion.PyMotion.set_motors`
:meth:`~openpibo.motion.PyMotion.set_speed`
:meth:`~openpibo.motion.PyMotion.set_acceleration`
:meth:`~openpibo.motion.PyMotion.set_init`

  파이보의 움직임을 제어하는 클래스.

  ``Motion`` 클래스와의 차이점은, 모터 컨트롤러와 직접 통신한다는 점입니다.

  example::

    from openpibo.motion import PyMotion

    pibo_pymotion = PyMotion()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  motor_range = [25,35,80,30,50,25,25,35,80,30]
  """
  PyMotion이 사용하는 Motor 별 가능한 범위입니다.
  0 -> 9 번 모터 순서이며, 절대값입니다. (모터0 이동가능범위: -25 ~ 25)
  """

  def __init__(self):
    """
    PyMotion 클래스를 초기화합니다.
    """

    self.dev = serial.Serial(port="/dev/ttyACM0", baudrate=115200)

  def set_motor(self, n, pos):
    """
    모터 1개를 특정 위치로 이동합니다.

    example::

      pibo_pymotion.set_motor(3, 30)

    :param int n: 모터 번호. 0~9 사이의 정수입니다.

    :param int pos: 모터 각도. -80~80 사이의 정수입니다.

      자세한 범위는 상단의 ``모터 제한 각도`` 를 참고해주세요.
    """

    if type(n) is not int:
      raise Exception(f'"{n}" must be integer type')

    if n < 0 or n > 9:
      raise Exception(f'"{n}" must be 0~9')

    if type(pos) is not int:
      raise Exception(f'"{pos}" must be integer type')

    if abs(pos) > PyMotion.motor_range[n]:
      raise Exception(f'Motor{n}\'s position must be -{PyMotion.motor_range[n]}~{PyMotion.motor_range[n]}')

    pos_val = (pos*10 + 1500)*4
    lsb = pos_val & 0x7f #7 bits for least significant byte
    msb = (pos_val >> 7) & 0x7f #shift 7 and take next 7 bits for msb
    cmd = chr(0x84) + chr(n) + chr(lsb) + chr(msb)
    self.dev.write(bytes(cmd,'latin-1'))

  def set_motors(self, positions): # pos array
    """
    전체 모터를 특정 위치로 이동합니다.

    example::

      pibo_pymotion.set_motors([20, 0, 0, 12, 10, 10, -10, -10, 0, 0])

    :param list positions: 0~9번 모터각도.

      **positions** 에 들어가는 범위는 상단의 ``모터 제한 각도`` 와 같습니다.
    """

    for n in range(len(positions)):
      if abs(positions[n]) > PyMotion.motor_range[n]:
        raise Exception(f'Motion{n}\'s position must be -{PyMotion.motor_range[n]}~{PyMotion.motor_range[n]}')

    positions_val = [(positions[i]*10+1500)*4 for i in range(len(d_lst))]
    cmd = chr(0x9F) + chr(10) + chr(0)
    for pos in positions_val:
      lsb = pos & 0x7f #7 bits for least significant byte
      msb = (pos >> 7) & 0x7f #shift 7 and take next 7 bits for msb
      cmd += chr(lsb) + chr(msb)
    self.dev.write(bytes(cmd,'latin-1'))

  def set_speed(self, n, speed):
    """
    모터 1개의 속도를 설정합니다.

    example::

      pibo_pymotion.set_speed(3, 255)

    :param int n: 모터 번호. 0~9 정수입니다.

    :param int speed: 모터 속도. 0~255 정수입니다.
    """

    if type(n) is not int:
      raise Exception (f'"{n}" must be integer type')

    if abs(n) > 9:
      raise Exception (f'"{n}" must be 0~9')

    if type(speed) is not int:
      raise Exception (f'"{speed}" must be integer type')

    if abs(speed) > 255:
      raise Exception (f'"{speed}" must be 0~255')

    lsb = speed & 0x7f #7 bits for least significant byte
    msb = (speed >> 7) & 0x7f #shift 7 and take next 7 bits for msb
    cmd = chr(0x87) + chr(n) + chr(lsb) + chr(msb)
    self.dev.write(bytes(cmd,'latin-1'))

  def set_acceleration(self, n, accel):
    """
    모터 1개의 가속도를 설정합니다.

    가속도를 설정하게 되면, 속도가 0에서 시작하여 설정된 속도까지 점점 빨라집니다.
    그리고 점점 느려지다가 종료 지점에서 속도가 0이 됩니다.

    example::

      pibo_pymotion.set_acceleration(3, 10)

    :param int n: 모터 번호. 0~9 정수입니다.

    :param int accel: 모터 가속도. 0~255 정수입니다.
    """

    if type(n) is not int:
      raise Exception (f'"{n}" must be integer type')

    if abs(n) > 9:
      raise Exception (f'"{n}" must be 0~9')

    if type(accel) is not int:
      raise Exception (f'"{accel}" must be integer type')

    if abs(accel) > 255:
      raise Exception (f'"{accel}" must be 0~255')

    lsb = accel & 0x7f #7 bits for least significant byte
    msb = (accel >> 7) & 0x7f #shift 7 and take next 7 bits for msb
    cmd = chr(0x89) + chr(n) + chr(lsb) + chr(msb)
    self.dev.write(bytes(cmd,'latin-1'))

  def set_init(self):
    """
    전체 모터를 초기 상태로 이동시킵니다.

    example::

      pibo_pymotion.set_init()

    """

    self.dev.write(bytes(chr(0xA2),'latin-1'))
