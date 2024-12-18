"""
mp3, wav 오디오 파일을 재생, 정지하고 마이크로 소리를 녹음합니다.

Class:
:obj:`~openpibo.audio.Audio`
"""

import os
from threading import Thread

HIGH = 1
LOW = 0

class Audio:
  """
Functions:
:meth:`~openpibo.audio.Audio.play`
:meth:`~openpibo.audio.Audio.stop`
:meth:`~openpibo.audio.Audio.mute`
:meth:`~openpibo.audio.Audio.record`

  mp3, wav 오디오 파일을 재생 및 정지합니다.

  example::

    from openpibo.audio import Audio

    audio = Audio()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """
  # volume: 0 ~ 100
  # filename: mp3/wav
  def __init__(self):
    os.system(f'gpio mode 7 out;gpio write 7 {HIGH}')

  def play(self, filename, volume=80, background=True, volume2=1.0):
    """
    mp3 또는 wav 파일을 재생합니다.

    example::

      audio.play('/home/pi/openpibo-files/audio/test.mp3', 80, True)

    :param str filename: 재생할 파일의 경로를 지정합니다.

      mp3와 wav 형식을 지원합니다.

    :param int volume: 음량을 설정합니다. (0~100)

    :param bool background: 오디오 파일을 백그라운드에서 실행할지 여부를 결정합니다.

      백그라운드에서 오디오가 재생되면, 오디오 재생되는 도중에 다른 명령어를 사용할 수 있습니다.

      * ``True``: 백그라운드에서 재생합니다. (default)
      * ``False``: 백그라운드에서 재생하지 않습니다.

    :param float volume2: 개별 음량을 조절합니다. (비율)
    """

    def play_thread(args):
      os.system(args)

    if not os.path.isfile(filename):
      raise Exception(f'"{filename}" does not exist')

    if not filename.split('.')[-1] in ['mp3', 'wav']:
      raise Exception(f'"{filename}" must be (mp3|wav)')

    if type(volume) is not int or (volume < 0 or volume > 100):
      raise Exception(f'"{volume}" is Number(0~100)')

    if type(background) is not bool:
      raise Exception(f'"{background}" is not bool')

    if type(volume2) is not float or (volume2 < 0.0 or volume2 > 3.0):
      raise Exception(f'"{volume2}" is float(0.0~1.5)')

    volume = int(volume/2) + 50 # 실제 50 - 100%로 설정, 0-50%는 소리가 너무 작음
    #cmd = f'amixer -q -c Headphones sset Headphone {volume}%;'
    cmd = f'amixer -q -c Headphones sset PCM {volume}%;'
    cmd += f'play -q -V1 -v {volume2} "{filename}"'

    if background:
      Thread(target=play_thread, args=(cmd,), daemon=True).start()
    else:
      os.system(cmd)

  def stop(self):
    """백그라운드에서 재생중인 오디오를 정지합니다.

    example::

      audio.stop()
    """

    os.system('sudo pkill play')

  def mute(self, value):
    """파이보를 무음모드로 만듭니다.

    인스턴스(pibo_audio)를 생성하면, 기본적으로 무음모드는 해제되어있습니다.

    무음모드에서는 ``play`` 메소드를 사용해도 소리가 출력되지 않습니다.

    example::

      audio.mute(True)

    :param bool value:

      * ``True``: 무음모드 설정.
      * ``False``: 무음모드 해제.
    """

    if type(value) != bool:
      raise Exception(f'"{value}" is not a bool')

    opt = LOW if value else HIGH
    os.system(f'gpio write 7 {opt}')

  def record(self, filename, timeout=5, verbose=True):
    """
    마이크로 소리를 녹음합니다.

    example::

      audio.record('/home/pi/test.wav', 5)

    :param str filename: 녹음한 파일이 저장 될 경로. ``wav`` 확장자를 사용합니다.

    :param int timeout: 녹음 시간(s)
    """

    if verbose == True:
      os.system(f'arecord -D plug:dmic_sv -c2 -r 16000 -f S32_LE -d {timeout} -t wav -q -vv -V streo stream.raw;sox stream.raw -c 1 -b 16 {filename};rm stream.raw')
    else:
      os.system(f'arecord -D dmic_sv -c2 -r 16000 -f S32_LE -d {timeout} -t wav -q stream.raw;sox stream.raw -q -c 1 -b 16 {filename};rm stream.raw')

if __name__ == "__main__":
  import time
  
  audio = Audio()
  audio.play("/home/pi/openpibo-files/audio/system/opening.mp3")
  time.sleep(3)
  audio.stop()
