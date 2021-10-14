"""
OLED Display에 문자나 이미지를 출력합니다.

Class:
:obj:`~openpibo.oled.Oled`
"""

from .modules.oled import ssd1306, board, busio, digitalio
from PIL import Image, ImageDraw, ImageFont
import PIL.ImageOps
import cv2
import numpy
import os

import openpibo_models
#current_path = os.path.dirname(os.path.abspath(__file__))

class Oled:
  """
Functions:
:meth:`~openpibo.oled.Oled.show`
:meth:`~openpibo.oled.Oled.clear`
:meth:`~openpibo.oled.Oled.set_font`
:meth:`~openpibo.oled.Oled.draw_text`
:meth:`~openpibo.oled.Oled.draw_image`
:meth:`~openpibo.oled.Oled.draw_data`
:meth:`~openpibo.oled.Oled.draw_rectangle`
:meth:`~openpibo.oled.Oled.draw_ellipse`
:meth:`~openpibo.oled.Oled.draw_line`
:meth:`~openpibo.oled.Oled.invert`

  파이보의 OLED를 통해 다양한 그림을 표현합니다.

  * 사진 보기
  * 문자 그리기
  * 도형 그리기

  그림을 그리면 인스턴스 변수 ``image`` 에 저장됩니다. 이를 ``show`` 메소드를 사용하여 oled 화면에 출력할 수 있습니다.

  본 class에서 문자 또는 그림을 그리는 행위는 인스턴스 변수 ``image`` 의 데이터를 변경시키는 것으로 정의합니다.

  example::

    from openpibo.oled import Oled

    pibo_oled = Oled()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    """
    Oled 클래스를 초기화
    """

    self.width = 128
    self.height = 64
    self.font_path = openpibo_models.filepath("KDL.ttf") # KoPub Dotum Light
    #self.font_path = current_path+"/data/models/KDL.ttf" # KoPub Dotum Light
    self.font_size = 10

    spi = busio.SPI(11, 10, 9)
    rst_pin = digitalio.DigitalInOut(board.D24) # any pin!
    cs_pin = digitalio.DigitalInOut(board.D8)    # any pin!
    dc_pin = digitalio.DigitalInOut(board.D23)    # any pin!

    self.oled = ssd1306.SSD1306_SPI(self.width, self.height, spi, dc_pin, rst_pin, cs_pin)
    self.font = ImageFont.truetype(self.font_path, self.font_size)
    self.image = Image.new("1", (self.width, self.height))
    self.oled.fill(0)
    self.oled.show()

  def show(self):
    """
    인스턴스 변수 ``image`` 를 oled 화면에 표시합니다.

    **이 메소드를 사용하지 않으면 그림을 그려도 oled 화면에 아무것도 출력되지 않습니다.**

    example::

      # 그림을 그린 후

      pibo_oled.show()
    """

    self.oled.image(self.image)
    self.oled.show()

  def clear(self):
    """
    인스턴스 변수 ``image`` 를 초기화 하고, oled 화면을 지웁니다.

    example::

      pibo_oled.clear()
    """

    self.image = Image.new("1", (self.width, self.height))
    self.oled.fill(0)
    self.oled.show()

  def set_font(self, filename=None, size=None):
    """
    ``draw_text`` 메소드에 사용할 폰트를 설정합니다.

    example::

      # 불러올 폰트의 경로가 /home/pi/mydata/font.ttf 라면,

      pibo_oled.set_font('/home/pi/mydata/font.ttf', 10)

    :param str filename: 폰트 파일 경로

      폰트 확장자는 **ttf** 와 **otf** 모두 지원합니다.

    :param int size: 폰트 사이즈

      단위는 픽셀 입니다. (default 10)
    """

    if filename == None:
      filename = self.font_path

    if size == None:
      size = self.font_size

    if not os.path.isfile(filename):
      raise Exception(f'"{filename}" does not exist')

    self.font = ImageFont.truetype(filename, size)

  def draw_text(self, points, text:str):
    """
    문자를 그립니다.(한글, 영어 지원)

    example::

      pibo_oled.draw_text((10, 10), '안녕하세요!')

    :param tuple(int, int) points: 문자열 좌측상단 좌표 (x, y)

    :param str text: 문자열 내용
    """

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 2:
      raise Exception(f'len({points}) must be 2')

    draw = ImageDraw.Draw(self.image)
    draw.text(points, text, font=self.font, fill=255)

  def draw_image(self, filename):
    """
    그림을 그립니다.

    **128x64** 크기의 **png** 확장자만 허용됩니다.

    example::

      pibo_oled.draw_image('/home/pi/openpibo-files/image/clear.png')

    :param str filename: 그림파일 경로
    """

    if not os.path.isfile(filename):
      raise Exception(f'"{filename}" does not exist') 

    self.image = Image.open(filename).convert('1')

  def draw_data(self, img):
    """
    numpy 이미지 데이터를 입력받아 이미지로 변환합니다.

    카메라 출력값이 numpy 형식이므로, 이를 oled화면에 띄우기 위해 사용됩니다.

    example::

      from openpibo.vision import Camera

      pibo_camera = Camera()
      img = pibo_camera.read(128, 64)

      pibo_oled.draw_data(img)
      pibo_oled.show()

    :param numpy.ndarray img: 이미지 객체
    """

    if type(img) is not numpy.ndarray:
      raise Exception('"img" must be image data from opencv.')

    self.image = Image.fromarray(img).convert('1')

  def draw_rectangle(self, points, fill=None):
    """
    직사각형을 그립니다.

    example::

      pibo_oled.draw_rectangle((10, 10, 80, 40), True)

    :param tuple points: 사각형의 좌측상단 좌표, 사각형의 우측하단 좌표 (x1, y1, x2, y2)

    :param bool fill:

      * ``True`` : 사각형 내부를 채웁니다.
      * ``False`` : 사각형 내부를 채우지 않습니다.
    """

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 4:
      raise Exception(f'len({points}) must be 4')

    if not fill in [None, True, False]:
      raise Exception(f'"{fill}" must be (None|True|False)')

    draw = ImageDraw.Draw(self.image)
    draw.rectangle(points, outline=1, fill=fill)

  def draw_ellipse(self, points, fill=None):
    """
    타원을 그립니다.

    example::

      pibo_oled.draw_ellipse((10, 10, 80, 40), False)

    :param tuple points: 타원에 외접하는 직사각형의 좌측상단 좌표, 우측하단 좌표 (x1, y1, x2, y2)

    :param bool fill:

      * ``True`` : 타원 내부를 채웁니다.
      * ``False`` : 타원 내부를 채우지 않습니다.
    """

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 4:
      raise Exception(f'len({points}) must be 4')

    if not fill in [None, True, False]:
      raise Exception(f'"{fill}" must be (None|True|False)')

    draw = ImageDraw.Draw(self.image)
    draw.ellipse(points, outline=1, fill=fill)

  def draw_line(self, points):
    """
    직선을 그립니다.

    example::

      pibo_oled.draw_line((30, 20, 60, 50))

    :param tuple points: 선의 시작 좌표, 선의 끝 좌표 (x1, y1, x2, y2)
    """

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 4:
      raise Exception(f'len({points}) must be 4')

    draw = ImageDraw.Draw(self.image)
    draw.line(points, fill=True)

  def invert(self):
    """
    그려진 이미지를 흑백 반전시킵니다.

    example::

      pibo_oled.invert()
    """
    self.image = self.image.convert("L")
    self.image = PIL.ImageOps.invert(self.image)
    self.image = self.image.convert("1")

