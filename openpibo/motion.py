"""
PIBO의 움직임을 제어합니다.

Class:
:obj:`~openpibo.motion.Motion`

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
:meth:`~openpibo.motion.Motion.set_motor`
:meth:`~openpibo.motion.Motion.set_motors`
:meth:`~openpibo.motion.Motion.set_speed`
:meth:`~openpibo.motion.Motion.set_speeds`
:meth:`~openpibo.motion.Motion.set_acceleration`
:meth:`~openpibo.motion.Motion.set_accelerations`
:meth:`~openpibo.motion.Motion.get_motion`
:meth:`~openpibo.motion.Motion.set_motion_raw`
:meth:`~openpibo.motion.Motion.set_motion`
:meth:`~openpibo.motion.Motion.set_mymotion`
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

  def get_motion(self, name=None, path=None):
    """
    모션 프로파일을 조회합니다.

    ``name`` 매개변수가 ``None`` 이면, 해당 모션 프로파일에 저장되어있는 모든 moiton 이름을 출력하고,
    ``None`` 이 아니면, 해당 이름의 모션에 대한 데이터를 출력합니다.

    example::

      pibo_motion.get_motion('forward1')

    :param str name: 동작 이름

      profile에 저장되어있는 동작의 이름입니다.

    :param str path: 사용할 모션 파일 경로

      모션 파일 경로입니다. 입력하지 않으면 기본 모션 파일을 사용합니다.

    :returns:

      * ``name == None`` 인 경우::

          # pibo_motion.get_motion()
          # pibo_motion.get_motion(path='/home/pi/mymotion.json')

          ['stop', 'stop_body', 'sleep', 'lookup', 'left', 'left_half',
          'right', 'right_half', 'forward1', 'forward2', ...]

      * ``name != None`` 인 경우::

          # pibo_motion.get_motion('stop')
          # pibo_motion.get_motion('stop', path='/home/pi/mymotion.json')

          {
            'comment': 'stop',
            'init_def': 1,
            'init': [0, 0, -70, -25, 0, 0, 0, 0, 70, 25]
          }
    """
    if path == None:
      return list(self.profile.keys()) if name == None else self.profile.get(name)
    elif os.path.isfile(path):
      with open(path, 'r') as f:
        result = json.load(f)
      return list(result.keys()) if name == None else result.get(name)
    else:
      raise Exception(f'"{path}" does not exist')
 
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
      self.set_motors(exe["init"], 1000)

    if "pos" not in exe:
      return

    time.sleep(1)
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

  def set_motion(self, name, cycle=1, path=None):
    """
    ``path`` 파일에 저장된 모션 프로파일의 모션을 실행합니다.
    ``path`` 를 설정하지 않으면, 기본 모션 프로파일이 적용됩니다.

    example::

      pibo_motion.set_motion('dance1')
      #pibo_motion.set_motion('dance1', path='/home/pi/mymotion.json')

    :param str name: 동작 이름

      모션 프로파일에 저장되어있는 동작의 이름입니다.

    :param int cycle:

      동작 반복 횟수

    :param str path:

      모션 파일 경로입니다. 입력하지 않으면 기본 모션 파일을 사용합니다.
    """

    if path == None:
      result = self.profile.get(name)
    elif os.path.isfile(path):
      with open(path, 'r') as f:
        result = json.load(f).get(name)
    else:
      raise Exception(f'"{path}" does not exist')

    if result == None:
      raise Exception(f'"{name}" does not exist in motion profile')
    return self.set_motion_raw(result, cycle)

  def set_mymotion(self, name, cycle=1):
    """
    ``/home/pi/mymotion.json`` 파일에 저장된 모션 프로파일의 모션을 실행합니다.

    example::

      pibo_motion.set_mymotion('test')

    :param str name: 동작 이름

      모션 프로파일에 저장되어있는 동작의 이름입니다.

    :param int cycle:

      동작 반복 횟수
    """

    return self.set_motion(name, cycle, path='/home/pi/mymotion.json')

  def stop(self):
    """
    수행 중인 동작을 정지합니다.

    example::

      # 동작을 수행 중일 때,
      pibo_motion.stop()
    """
    self.stopped = True
