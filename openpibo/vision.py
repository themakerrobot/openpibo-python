"""
`OpenCV` 라이브러리를 활용한 영상처리, 인공지능 비전 기술을 사용합니다.

Class:
:meth:`~openpibo.vision.vision_api`
:obj:`~openpibo.vision.Camera`
:obj:`~openpibo.vision.Face`
:obj:`~openpibo.vision.Detect`
:obj:`~openpibo.vision.TeachableMachine`
"""

import cv2,dlib,requests
import os,pickle,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tflite_runtime.interpreter import Interpreter
from pyzbar import pyzbar
from .modules.pose.movenet import Movenet
from .modules.pose.utils import visualize
from .modules.card.decode_card import get_card

import openpibo_models
import openpibo_face_models
import openpibo_dlib_models
import openpibo_detect_models

def vision_api(mode, image, params={}):
  """
  인공지능 비전 API를 호출합니다.

  example::

    from openpibo.vision import vision_api, Camera

    camera = Camera()

    res = vision_api('caption/caption', camera.read(), {})

  :param str mode: 호출할 비전 API

  :param str/numpy.ndarray image: 표시할 이미지 파일 경로 / 데이터(cv2)

  :returns: ``Json`` 타입 결과의 데이터

    example::
    
      { 'type': 'caption', 'result': 'ok', 
        'data': {
          caption:  "사람에게 로봇을 과시하는 사람", 
          caption_en:  "a person showing off a robot to a person",
          raw:  [
            "a person showing off a robot to a person",
            "a robot that is sitting on top of a table",
            "a very cute white robot that is sitting in front of a table"
          ]
        }
      }
  """

  if type(image) is np.ndarray:
    return requests.post(f"https://o-vapi.circul.us/{mode}", files={'uploadFile':cv2.imencode('.jpg', image)[1].tobytes()}, params=params).json()
  else:
    return requests.post(f"https://o-vapi.circul.us/{mode}", files={'uploadFile':open(image, 'rb')}, params=params).json()

class Camera:
  """
Functions:
:meth:`~openpibo.vision.Camera.imread`
:meth:`~openpibo.vision.Camera.read`
:meth:`~openpibo.vision.Camera.imshow_to_ide`
:meth:`~openpibo.vision.Camera.resize`
:meth:`~openpibo.vision.Camera.rotate`
:meth:`~openpibo.vision.Camera.imwrite`
:meth:`~openpibo.vision.Camera.rectangle`
:meth:`~openpibo.vision.Camera.circle`
:meth:`~openpibo.vision.Camera.line`
:meth:`~openpibo.vision.Camera.putText`
:meth:`~openpibo.vision.Camera.stylization`
:meth:`~openpibo.vision.Camera.detailEnhance`
:meth:`~openpibo.vision.Camera.pencilSketch`
:meth:`~openpibo.vision.Camera.flip`

  파이보의 카메라를 제어합니다.

  * 사진 촬영, 읽기, 쓰기 등 카메라 기본 기능을 사용할 수 있습니다.
  * 이미지에 도형/글자를 추가할 수 있습니다.
  * 이미지를 변환할 수 있습니다.

  example::

    from openpibo.vision import Camera

    camera = Camera()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self, cam=0):
    """
    Camera 클래스를 초기화합니다.
    """

    os.system('v4l2-ctl -c vertical_flip=1,horizontal_flip=1,white_balance_auto_preset=3')
    self.cap = cv2.VideoCapture(cam)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # for opencv buffering issue

  def imread(self, filename):
    """
    이미지 파일을 읽습니다.

    example::

      camera.imread('/home/pi/openpibo-files/data/image/clear.png')

    :param str filename: 사용할 이미지 파일

    :returns: ``numpy.ndarray`` 타입 이미지 객체

      example::

        array([[[0, 0, 0],
                [0, 0, 0],
                ...,
                [0, 0, 0],
                [0, 0, 0]],

              [[0, 0, 0],
                [0, 0, 0],
                ...,
                [0, 0, 0],
                [0, 0, 0]],

              ...,

              [[0, 0, 0],
                [0, 0, 0],
                ...,
                [0, 0, 0],
                [0, 0, 0]]], dtype=uint8)
    """

    return cv2.imread(filename)

  def read(self):
    """
    카메라를 통해 이미지를 촬영합니다.

    해상도 변경 시 이미지가 깨질 수 있으므로, 기본 해상도를 권장합니다.

    example::

      camera.read()

    :returns: ``numpy.ndarray`` 타입 이미지 객체
    """

    # for opencv buffering issue
    self.cap.grab()
    self.cap.grab()
    return self.cap.read()[1]

  def imshow_to_ide(self, img, ratio=0.25):
    """
    이미지 파일을 Web IDE에 출력합니다.

    example::

      img = camera.read()
      camera.imshow_to_ide(img)

    :param numpy.ndarray img: 이미지 객체

    :param float ratio: 이미지 사이즈 변환 비율 (다수 동시 사용시, 네트워크 부하)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    # curl -X 'POST' -so /dev/null 'http://0.0.0.0:50000/show' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'data=@{filename};type=image/png'
    requests.post('http://0.0.0.0:50000/show', headers={'accept': 'application/json'}, files={'data': ('filename', cv2.imencode('.jpg', img)[1].tobytes(), 'image/png')})

  def resize (self, img, w, h):
    """
    Opencv 이미지의 크기를 변환합니다.

    example::

      img = camera.read()
      camera.resize(img, 128, 64)

    :param numpy.ndarray img: 이미지 객체

    :param int w: 변환될 이미지의 가로 크기입니다. (픽셀 단위)

    :param int h: 변환될 이미지의 세로 크기입니다. (픽셀 단위)

    :returns: 크기 변환 후의 이미지 객체
    """

    return cv2.resize(img, (w, h))

  def rotate(self, img, degree=10, ratio=0.9):
    """
    이미지를 회전시킵니다.

    example::

      img = camera.read()
      camera.rotate(img, 10, 0.9)

    :param numpy.ndarray img: 이미지 객체

    :param int degree: 회전할 각도

    :param float ratio: 축소 또는 확대할 비율

    :returns: 회전한 ``numpy.ndarray`` 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(degree) is not int or abs(degree) >= 360:
      raise Exception(f'{degree} must be integer type and -360~360')

    if type(ratio) is not float or ratio >= 1.0:
      raise Exception(f'"{ratio} must be float type and 0~1.0')

    rows, cols = img.shape[0:2]
    op = cv2.getRotationMatrix2D((cols/2,rows/2), degree, ratio)
    return cv2.warpAffine(img, op, (cols,rows))

  def imwrite(self, filename, img):
    """
    이미지를 파일로 저장합니다.

    example::

      img = camera.read()
      camera.imwrite('/home/pi/image.jpg', img)

    :param str filename: 저장할 파일 경로

      확장자는 jpg 또는 png를 사용할 수 있습니다.

    :param numpy.ndarray img: 저장할 이미지 객체
    """

    return cv2.imwrite(filename, img)

  def rectangle(self, img, p1, p2, colors=(255,255,255), tickness=1):
    """
    이미지에 직사각형을 그립니다.

    example::

      img = camera.read()
      camera.rectangle(img, (10, 10), (300, 200), (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param tuple(int, int) p1: 좌측상단 좌표 (x, y)

    :param tuple(int, int) p2: 우측하단 좌표 (x, y)

    :param tuple(int, int, int) colors: RGB 값 (r, g, b) or 16진수 값 '#ffffff'

    :param int tickness: 사각형 모서리의 두께 (픽셀 단위)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(p1) is not tuple:
      raise Exception(f'"{p1}" must be tuple type')

    if len(p1) != 2:
      raise Exception(f'len({p1}) must be 2')

    if type(p2) is not tuple:
      raise Exception(f'"{p2}" must be tuple type')

    if len(p2) != 2:
      raise Exception(f'len({p2}) must be 2')

    if type(colors) is str:
      colors = (int(colors[5:7], 16), int(colors[3:5], 16), int(colors[1:3], 16))

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.rectangle(img, p1, p2, colors, tickness)

  def circle(self, img, p, r, colors=(255,255,255), tickness=1):
    """
    이미지에 직사각형을 그립니다.

    example::

      img = camera.read()
      camera.circle(img, (10, 10), (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param tuple(int, int) p: 좌측상단 좌표 (x, y)

    :param int r: 반지름

    :param tuple(int, int, int) colors: RGB 값 (r, g, b) or 16진수 값 '#ffffff'

    :param int tickness: 사각형 모서리의 두께 (픽셀 단위)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(p) is not tuple:
      raise Exception(f'"{p}" must be tuple type')

    if len(p) != 2:
      raise Exception(f'len({p}) must be 2')

    if type(r) is not int:
      raise Exception(f'len({r}) must be Integer type')

    if type(colors) is str:
      colors = (int(colors[5:7], 16), int(colors[3:5], 16), int(colors[1:3], 16))

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.circle(img, p, r, colors, tickness)

  def line(self, img, p1, p2, colors=(255,255,255), tickness=1):
    """
    이미지에 직선을 그립니다.

    example::

      img = camera.read()
      camera.line(img, (10, 10), (300, 200), (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param tuple(int, int) p1: 시작 좌표 (x, y)

    :param tuple(int, int) p2: 끝 좌표 (x, y)

    :param tuple(int, int, int) colors: RGB 값 (r, g, b) or 16진수 값 '#ffffff'

    :param int tickness: 선의 두께 (픽셀 단위)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(p1) is not tuple:
      raise Exception(f'"{p1}" must be tuple type')

    if len(p1) != 2:
      raise Exception(f'len({p1}) must be 2')

    if type(p2) is not tuple:
      raise Exception(f'"{p2}" must be tuple type')

    if len(p2) != 2:
      raise Exception(f'len({p2}) must be 2')

    if type(colors) is str:
      colors = (int(colors[5:7], 16), int(colors[3:5], 16), int(colors[1:3], 16))

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.line(img, p1, p2, colors, tickness)

  def putTextPIL(self, img, text, points, size=30, colors=(255,255,255)):
    """
    이미지에 문자를 입력합니다. (한/영 가능 - pillow 이용)

    example::

      img = camera.read()
      new_img = camera.putTextPIL(img, '안녕하세요.', (15, 10), 30, (255, 255, 255))

    :param numpy.ndarray img: 이미지 객체

    :param str text: 표시할 문자열

    :param tuple(int, int) points: 텍스트 블록 좌측상단 좌표 (x, y)

    :param int size: 표시할 글자의 크기

    :param tuple(int, int, int) colors: 글자 색깔 RGB 값 (b, g, r) or 16진수 값 '#ffffff'

    """
    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 2:
      raise Exception(f'len({points}) must be 2')

    if type(colors) is str:
      colors = (int(colors[5:7], 16), int(colors[3:5], 16), int(colors[1:3], 16))

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    font = ImageFont.truetype(openpibo_models.filepath("KDL.ttf"), size)
    pil = Image.fromarray(img)  # CV to PIL
    ImageDraw.Draw(pil).text(points, text, font=font, fill=colors)  # putText
    return np.array(pil)  # PIL to CV

  def putText(self, img, text, points, size=1, colors=(255,255,255), tickness=1):
    """
    이미지에 문자를 입력합니다. (영어만 가능)

    example::

      img = camera.read()
      new_img = camera.putText(img, 'hello', (15, 10), 10, (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param str text: 표시할 문자열

    :param tuple(int, int) points: 텍스트 블록 좌측하단 좌표 (x, y)

    :param int size: 표시할 글자의 크기

    :param tuple(int, int, int) colors: 글자 색깔 RGB 값 (r, g, b) or 16진수 값 '#ffffff'

    :param int tickness: 글자 두께
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 2:
      raise Exception(f'len({points}) must be 2')

    if type(colors) is str:
      colors = (int(colors[5:7], 16), int(colors[3:5], 16), int(colors[1:3], 16))

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.putText(img, text, points, cv2.FONT_HERSHEY_SIMPLEX, size, colors, tickness)

  def stylization(self, img, sigma_s=100, sigma_r=0.5):
    """
    만화 이미지로 변환합니다. (opencv api) low speed

    example::

      img = camera.read()
      camera.stylization(img)

    :param numpy.ndarray img: 이미지 객체

    :param float sigma_s: 이미지의 blur 보존 정도 (1-200)

    :param float sigma_r: 이미지의 Edge 적용 정도 (0-1)

    :returns: 변환 후의 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)

  def detailEnhance(self, img, sigma_s=100, sigma_r=0.05):
    """
    만화 이미지로 변환합니다. (opencv api)

    example::

      img = camera.read()
      camera.stylization(img)

    :param numpy.ndarray img: 이미지 객체

    :param float sigma_s: 이미지의 blur 보존 정도 (1-200)

    :param float sigma_r: 이미지의 Edge 적용 정도 (0-1)

    :returns: 변환 후의 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

  def pencilSketch(self, img, sigma_s=100, sigma_r=0.2, shade_factor=0.018):
    """
    스케치 이미지로 변환합니다.

    example::

      img = camera.read()
      camera.sketchize(img)

    :param numpy.ndarray img: 이미지 객체

    :param float sigma_s: 이미지의 blur 보존 정도 (1-200)

    :param float sigma_r: 이미지의 Edge 적용 정도 (0-1)

    :param float shade_factor: 이미지의 밝기 정도 (0-0.1)

    :returns: 변환 후의 이미지 객체(grayscale), 변환 후의 이미지 객체(bgr)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.pencilSketch(img, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)

  def edgePreservingFilter(self, img, flags=1, sigma_s=60, sigma_r=0.4):
    """
    흐림 이미지로 변환합니다.

    example::

      img = camera.read()
      camera.edgePreservingFilter(img)

    :param numpy.ndarray img: 이미지 객체

    :param int flags: 필터 종류 1 (RECURS_FILTER) or 2 (NORMCONV_FILTER)

    :param float sigma_s: 이미지의 blur 보존 정도 (1-200)

    :param float sigma_r: 이미지의 Edge 적용 정도 (0-1)

    :returns: 변환 후의 이미지 객체(grayscale), 변환 후의 이미지 객체(bgr)
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.edgePreservingFilter(img, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)

  def flip(self, img, flags=1):
    """
    상하/좌우 대칭 이미지로 변환합니다

    example::

      img = camera.read()
      camera.flip(img, 1)

    :param numpy.ndarray img: 이미지 객체

    :param int flags: 0: 상하 대칭, 1: 좌우 대칭, -1: 상하/좌우 대칭

    :returns: 변환 후, 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.flip(img, flags)

class Face:
  """
Functions:
:meth:`~openpibo.vision.Face.detect`
:meth:`~openpibo.vision.Face.get_ageGender`
:meth:`~openpibo.vision.Face.get_age`
:meth:`~openpibo.vision.Face.get_gender`
:meth:`~openpibo.vision.Face.init_db`
:meth:`~openpibo.vision.Face.train_face`
:meth:`~openpibo.vision.Face.delete_face`
:meth:`~openpibo.vision.Face.recognize`
:meth:`~openpibo.vision.Face.get_db`
:meth:`~openpibo.vision.Face.save_db`
:meth:`~openpibo.vision.Face.load_db`

  얼굴과 관련된 다양한 기능을 수행하는 클래스입니다. 다음 기능을 수행할 수 있습니다.

  * 얼굴을 탐색합니다.
  * 얼굴을 학습/저장/삭제합니다.
  * 학습된 얼굴을 인식합니다.
  * 얼굴로 나이/성별/감정을 추정합니다.

  :얼굴 데이터베이스: 인스턴스 변수 **facedb** 를 의미하며, 여기에서 얼굴 데이터를 등록하고 불러오고 삭제합니다.

    얼굴 데이터베이스의 포맷은 이중 list ``[[], []]`` 이며, 첫 번째 list에는 얼굴의 이름이, 두 번째 list에는 학습된 얼굴 데이터가 인코딩되어 들어갑니다.

    또한 파일로 저장하여 인스턴스가 삭제된 후에도 얼굴 정보를 남겨둘 수 있습니다.

  example::

    from openpibo.vision import Face

    face = Face()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.facedb = [[],[]]
    self.threshold = 0.4
    self.age_class = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    self.gender_class = ['Male', 'Female']
    self.agenet = cv2.dnn.readNetFromCaffe(
                    openpibo_face_models.filepath("deploy_age.prototxt"),
                    openpibo_face_models.filepath("age_net.caffemodel")
                )
    self.gendernet = cv2.dnn.readNetFromCaffe(
                    openpibo_face_models.filepath("deploy_gender.prototxt"),
                    openpibo_face_models.filepath("gender_net.caffemodel")
                )
    #self.face_detector = cv2.CascadeClassifier(openpibo_face_models.filepath("haarcascade_frontalface_default.xml"))
    self.face_detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(openpibo_dlib_models.filepath("shape_predictor_68_face_landmarks.dat"))
    self.face_encoder = dlib.face_recognition_model_v1(openpibo_dlib_models.filepath("dlib_face_recognition_resnet_model_v1.dat"))

  def detect_face(self, img):
    """
    얼굴을 탐색합니다.

    example::

      img = camera.read()
      face.detect_face(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 인식된 얼굴들의 (x, y, w, h) 배열 입니다.

      list 타입으로, 이미지 하나에 얼굴이 여러 개 인식된 경우 인식된 얼굴의 좌표가 모두 입력됩니다.

      example::

        [(10, 10, 40, 50), (120, 30, 160, 70), (130, 140, 200, 260)]
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    return [(d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()) for d in self.face_detector(img)]
    #return self.face_detector.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5) # [(x,y,w,h), ...]

  def landmark_face(self, img):
    """
    얼굴의 랜드마크를 탐색합니다.

    example::

      img = camera.read()
      face.landmark_face(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"data": 인식한 결과, "img": landmark를 반영한 이미지 }``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    res = []
    rects = self.face_detector(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for rect in rects:
      #rect = dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
      shape = self.predictor(gray, rect)

      coords = np.zeros((shape.num_parts, 2), dtype="int")
      for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        cv2.circle(img, (coords[i][0], coords[i][1]), 2, (50, 200, 50), -1)
        cv2.putText(img, str(i+1), (coords[i][0]-10, coords[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 200, 50), 1)

      res.append(coords)

    return {"data":res, "img":img}

  def get_ageGender(self, img, item):
    """
    얼굴의 나이, 성별을 추정합니다.

    example::

      img = camera.read()
      items = face.detect_face(img)
      item = items[0] # item은 items 중 하나
      face.get_ageGender(img, item)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray item: 얼굴의 좌표 (x, y, w, h)

    :returns: ``{"age": 나이, "gender": 성별}``

      * age: 나이의 범위를 tuple() 형태로 출력한다.

        ex) (15, 20) # 15살에서 20살 정도

      * gender: ``male`` / ``female``

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if len(item) != 4:
      raise Exception('"item" must be [x,y,w,h]')

    x, y, w, h = item
    face_img = img[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict gender
    self.gendernet.setInput(blob)
    gender_preds = self.gendernet.forward()
    gender = self.gender_class[gender_preds[0].argmax()]

    # predict age
    self.agenet.setInput(blob)
    age_preds = self.agenet.forward()
    age = self.age_class[age_preds[0].argmax()]

    return {"age":age, "gender":gender}

  def get_age(self, img, item):
    """
    얼굴의 나이를 추정합니다.

    example::

      img = camera.read()
      items = face.detect_face(img)
      item = items[0] # item는 items중 하나
      face.get_age(img, item)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray item: 얼굴의 좌표 (x, y, w, h)

    :returns: ``나이 범위`` ``Raw데이터``

      * age: 나이의 범위를 tuple() 형태로 출력한다.

        ex) (15, 20) # 15살에서 20살 정도

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if len(item) != 4:
      raise Exception('"item" must be [x,y,w,h]')

    x, y, w, h = item
    face_img = img[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict age
    self.agenet.setInput(blob)
    age_preds = self.agenet.forward()
    age = self.age_class[age_preds[0].argmax()]

    return age, age_preds[0]

  def get_gender(self, img, item):
    """
    얼굴의 성별을 추정합니다.

    example::

      img = camera.read()
      items = face.detect_face(img)
      item = items[0] # item는 items중 하나
      face.get_gender(img, item)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray item: 얼굴의 좌표 (x, y, w, h)

    :returns: ``성별`` ``Raw데이터``

      * gender: ``male`` / ``female``

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if len(item) != 4:
      raise Exception('"item" must be [x,y,w,h]')

    x, y, w, h = item
    face_img = img[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict gender
    self.gendernet.setInput(blob)
    gender_preds = self.gendernet.forward()
    gender = self.gender_class[gender_preds[0].argmax()]

    return gender, gender_preds[0]

  def init_db(self):
    """
    얼굴 데이터베이스를 초기화합니다.

    초기화된 데이터베이스는 빈 이중 list ``[[], []]`` 입니다.

    example::

      face.init_db()
    """

    self.facedb = [[], []]

  def train_face(self, img, item, name):
    """
    얼굴을 학습하여 얼굴 데이터베이스에 저장합니다.

    example::

      img = camera.read()
      items = face.detect_face(img)
      item = items[0] # item는 items중 하나
      face.train_face(img, item, 'honggildong')

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray item: 디텍팅한 얼굴의 사각형 좌측상단, 우측하단 포인트 (x1, y1, x2, y2)

    :param str name: 디텍팅한 얼굴에 붙일 이름
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    if len(item) != 4:
      raise Exception('"item" must be [x,y,w,h]')

    x,y,w,h = item
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = self.predictor(gray, rect)
    face_encoding = np.array(self.face_encoder.compute_face_descriptor(img, shape, 1))

    self.facedb[0].append(name)
    self.facedb[1].append(face_encoding)
    #cv2.imwrite(self.data_path+"/{}.jpg".format(name), img[y+3:y+h-3, x+3:x+w-3]);

  def delete_face(self, name):
    """
    등록된 얼굴을 삭제합니다.

    example::

      face.delete_face('honggildong')

    :param str name: 삭제할 얼굴의 이름

    :returns: ``True`` / ``False``
    """

    ret = name in self.facedb[0]
    if ret == True:
      idx = self.facedb[0].index(name)
      #os.remove(self.data_path +"/" + name + ".jpg")
      for item in self.facedb:
        del item[idx]

    return ret

  def recognize(self, img, item):
    """
    등록된 얼굴을 인식합니다.

    example::

      img = camera.read()
      items = face.detect_face(img)
      item = items[0] # item는 items중 하나
      face.recognize(img, item)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray item: 얼굴의 좌표 (x, y, w, h)

    :returns: ``{"name": 이름, "score": 오차도}``

      얼굴이 비슷할수록 오차도가 낮게 측정됩니다.

      오차도가 0.4 이하일 때 동일인으로 판정합니다.
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    if len(item) != 4:
      raise Exception('"item" must be [x,y,w,h]')

    if len(self.facedb[0]) < 1:
      return {"name":"Guest", "score":0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data={"name":"Guest", "score":0}
    (x,y,w,h) = item
    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    shape = self.predictor(gray, rect)
    face_encoding = np.array(self.face_encoder.compute_face_descriptor(img, shape, 1))
    matches = []
    matches = list(np.linalg.norm(self.facedb[1] - face_encoding, axis=1))
    data["score"] = round(min(matches), 2)

    if min(matches) < self.threshold:
      data["name"] = self.facedb[0][matches.index(min(matches))]
    return data

  def get_db(self):
    """
    사용 중인 얼굴 데이터베이스를 확인합니다.

    example::

      face.get_db()

    :returns: **facedb** (``list(list, list)`` 타입)

      example::

        [
          ['honggildong'],
          [array([-0.06423206,  0.12474005,  0.0511112 , -0.05676335, -0.07211345,
                  -0.03123881, -0.04119622, -0.12800875,  0.11717855, -0.11079554,
                   0.22952782, -0.02007426, -0.17457265, -0.13562854, -0.04972655,
                   0.15810637, -0.12785575, -0.16479518, -0.07002968, -0.00208595,
                   0.169218  ,  0.03144928, -0.01074579,  0.04103286, -0.09245337,
                  ...
                  -0.00706697,  0.06025593, -0.0049719 ])]
        ]
    """

    return self.facedb

  def save_db(self, filename):
    """
    얼굴 데이터베이스를 파일로 저장합니다.

    example::

      face.save_db('/home/pi/facedb')

    :param str filename: 저장할 얼굴 데이터베이스 파일의 경로입니다.
    """

    with open(filename, "w+b") as f:
      pickle.dump(self.facedb, f)

  def load_db(self, filename):
    """
    얼굴 데이터베이스 파일을 불러옵니다.

    example::

      face.load_db('/home/pi/facedb')

    :param str filename: 불러 올 ``facedb`` 파일의 경로입니다.
    """

    if not os.path.isfile(filename):
      raise Exception('"{filename}" does not exist')

    with open(filename, "rb") as f :
      self.facedb = pickle.load(f)

class Detect:
  """
Functions:
:meth:`~openpibo.vision.Detect.detect_object`
:meth:`~openpibo.vision.Detect.detect_qr`
:meth:`~openpibo.vision.Detect.detect_pose`
:meth:`~openpibo.vision.Detect.analyze_pose`
:meth:`~openpibo.vision.Detect.classify_image`
:meth:`~openpibo.vision.Detect.object_tracker_init`
:meth:`~openpibo.vision.Detect.track_object`
:meth:`~openpibo.vision.Detect.detect_marker`

  인식과 관련된 다양한 기능을 사용할 수 있는 클래스입니다.

  * 90개 class 안에서의 객체 인식 (MobileNet V2)
  * QR/바코드 인식 (pyzbar)
  * Pose 인식
  * 이미지 분류

  example::

    from openpibo.vision import Detect

    detect = Detect()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.object_class = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                         'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                         'None', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'None',
                         'backpack', 'umbrella', 'None', 'None', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'None', 'wine glass', 'cup',
                         'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                         'potted plant', 'bed', 'None', 'dining table', 'None', 'None', 'toilet', 'None', 'tv',
                         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                         'toaster', 'sink', 'refrigerator', 'None', 'book', 'clock', 'vase', 'scissors',
                         'teddy bear', 'hair drier'] 
    
    self.mobilenet = cv2.dnn.readNet(
                        openpibo_detect_models.filepath("frozen_inference_graph.pb"),
                        openpibo_detect_models.filepath("ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
                    )
    self.pose_detector = Movenet(openpibo_detect_models.filepath("movenet_lightning.tflite"))

    with open(openpibo_detect_models.filepath("efficientnet_labels.txt"), 'r') as f:
      self.cls_class_names = [item.strip() for item in f.readlines()]
    self.cls_interpreter = Interpreter(model_path=openpibo_detect_models.filepath("efficientnet_lite3.tflite"))
    self.cls_interpreter.allocate_tensors()
    self.cls_input_details = self.cls_interpreter.get_input_details()
    self.cls_output_details = self.cls_interpreter.get_output_details()
    self.camera_matrix = np.array([
      [1.42068235e+03,0.00000000e+00,9.49208512e+02],
      [0.00000000e+00,1.37416685e+03,5.39622051e+02],
      [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    self.distortion_coeff = np.array([1.69926613e-01,-7.40003491e-01,-7.45655262e-03,-1.79442353e-03, 2.46650225e+00])
    self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    self.parameters = cv2.aruco.DetectorParameters_create()

  def detect_object(self, img):
    """
    이미지 안의 객체를 인식합니다. (아래 class의 사물 인식 가능)

    인식 가능한 사물은 다음과 같습니다::

      'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
      'None', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'None',
      'backpack', 'umbrella', 'None', 'None', 'handbag', 'tie', 'suitcase', 'frisbee',
      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'None', 'wine glass', 'cup',
      'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
      'potted plant', 'bed', 'None', 'dining table', 'None', 'None', 'toilet', 'None', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
      'toaster', 'sink', 'refrigerator', 'None', 'book', 'clock', 'vase', 'scissors',
      'teddy bear', 'hair drier' 

    example::

      img = camera.read()
      detect.detect_object(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"name":이름, "score":정확도, "position":사물좌표(startX, startY, endX, endY)}``

      * score는 0~100 사이의 float 값 입니다.
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    data = []
    class_ids = []
    scores = []
    boxes = []
    boxes_nms = []    
    img_h, img_w = img.shape[:2]
    self.mobilenet.setInput(cv2.dnn.blobFromImage(img, size=(300,300), swapRB=True))
    output = self.mobilenet.forward()

    for detection in output[0, 0, :, :]:
      if detection[2] > .5:
        x1 = max(10, int(detection[3] * img_w))
        y1 = max(10, int(detection[4] * img_h))
        x2 = min(img_w + 10, int(detection[5] * img_w))
        y2 = min(img_h + 10, int(detection[6] * img_h))
        class_ids.append(int(detection[1]))
        scores.append(float(detection[2]))
        boxes_nms.append((x1, y1, x2-x1, y2-y1)) # NMSBoxes box: x,y,w,h
        boxes.append((x1, y1, x2, y2))

    idxs = cv2.dnn.NMSBoxes(boxes_nms, scores, .5, .4)
    if len(idxs) > 0:
      for i in idxs.flatten():
        data.append({"name":self.object_class[class_ids[i]], "score":int(scores[i]*100), "position":boxes[i]})
    return data

  def detect_qr(self, img):
    """
    이미지 안의 QR코드 및 바코드를 인식합니다.

    example::

      img = camera.read()
      detect.detect_qr(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"data": 내용, "type": 바코드 / QR코드, "position":(startX,startY,endX,endY)}``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    barcodes = pyzbar.decode(img)
    if len(barcodes) > 0:
      x,y,w,h = barcodes[0].rect
      _type = barcodes[0].type
      _data = barcodes[0].data.decode("utf-8")

      res = get_card(_data)
      if res != None:
        _type, _data = "CARD", res
      return {"data":_data, "type":_type, "position":(x,y,x+w,y+h)}
    else:
      return {"data":"", "type":"", "position":None}

  def detect_pose(self, img):
    """
    이미지 안의 Pose를 인식합니다.

    example::

      img = camera.read()
      detect.detect_pose(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"data": 인식한 결과, "img": Pose를 반영한 이미지 }``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    list_persons = [self.pose_detector.detect(img)]
    img = visualize(img, list_persons)
    return {"data":list_persons, "img":img}

  def analyze_pose(self, data):
    """
    detect_pose 함수의 결과 값을 분석합니다.

    example::

      img = camera.read()
      result = detect.detect_pose(img)
      detect.analyze_pose(result)

    :param dict data: 이미지 객체

    :returns: ``인식한 포즈 리스트 ['left_hand_up', 'right_hand_up', 'clap']``
    """

    def distance(p1, p2):
      return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0,1,2,3,4
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 5,6,7,8,9,10
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 11,12,13,14,15,16

    res = []
    data = data['data'][0].keypoints

    if data[LEFT_WRIST].coordinate.y < data[LEFT_ELBOW].coordinate.y:
      res.append("left_hand_up")
    if data[RIGHT_WRIST].coordinate.y < data[RIGHT_ELBOW].coordinate.y:
      res.append("right_hand_up")
    if distance(data[LEFT_WRIST].coordinate, data[RIGHT_WRIST].coordinate) <  75:
      res.append("clap")

    return res

  def classify_image(self, img, k=5):
    """
    이미지를 분류합니다.

    example::

      img = camera.read()
      detect.classify_image(img)

    :param numpy.ndarray img: 이미지 객체

    :param int k: 가져올 결과 항목 개수

    :returns: ``top_k 리스트 [{"score":정확도, "name":이름}, ...]``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    height = self.cls_input_details[0]['shape'][1]
    width = self.cls_input_details[0]['shape'][2]
    img = cv2.resize(img, (height, width))
    image = Image.fromarray(img)

    input_data = np.expand_dims(image, axis=0).reshape(-1, height, width, 3)

    # feed data to input tensor and run the interpreter
    self.cls_interpreter.set_tensor(self.cls_input_details[0]['index'], input_data)
    self.cls_interpreter.invoke()

    preds = self.cls_interpreter.get_tensor(self.cls_output_details[0]['index'])
    preds = np.squeeze(preds)
    top_k = preds.argsort()[-k:][::-1]

    return [{"score":int(100*float(preds[i]/255.0)), "name":self.cls_class_names[i]} for i in top_k]

  def object_tracker_init(self, img, p):
    """
    이미지 안의 사물 트래커를 설정합니다.

    example::

      img = camera.read()
      detect.object_tracker_init(img)

    :param numpy.ndarray img: 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    x1,y1,x2,y2 = p
    tracker = dlib.correlation_tracker()
    tracker.start_track(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dlib.rectangle(x1,y1,x2,y2))
    return tracker

  def track_object(self, tracker, img):
    """
    이미지 안의 사물 트래커를 설정합니다.

    example::

      img = camera.read()
      tracker = detect.object_tracker_init(img, (10,10,100,100))
      tracker, position = detect.track_object(tracker, img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"tracker": 업데이트된 tracker, "position": 업데이트된 사물 위치 }``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    tracker.update(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pos = tracker.get_position()

    x1 = int(pos.left())
    y1 = int(pos.top())
    x2 = int(pos.right())
    y2 = int(pos.bottom())
    return {'tracker':tracker, 'position':(x1, y1, x2, y2)}

  def detect_marker(self, img, marker_length=2):
    """
    이미지 안의 마커를 인식합니다. # cv2.aruco.DICT_4X4_50

    example::

      img = camera.read()
      result = detect.detect_marker(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``[{"id": 마커번호, "center":  }, ...]``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    marker_length /= 250
    corners, ids, _ = cv2.aruco.detectMarkers(img, self.dictionary, parameters=self.parameters)
    res = []
    if len(corners) > 0:
      img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
      ids = ids.flatten()

      for (corner, markerID) in zip(corners, ids):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, marker_length, self.camera_matrix, self.distortion_coeff)
        (topLeft, topRight, bottomRight, bottomLeft) = corner.reshape((4, 2))

        topRight = (int(topRight[0]), int(topRight[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        distance = round(tvec[0][0][2] * 100, 1) #[cm]

        # cv2.line(img, topLeft, topRight, (255, 0, 0), 4)
        # cv2.line(img, topRight, bottomRight, (255, 0, 0), 4)
        # cv2.line(img, bottomRight, bottomLeft, (255, 0, 0), 4)
        # cv2.line(img, bottomLeft, topLeft, (255, 0, 0), 4)
        # cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
        # cv2.putText(img, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        res.append({"id":markerID, "center": (cX, cY), "box": [topLeft, topRight, bottomRight, bottomLeft], "distance":distance})

    return {"data":res, "img":img}

class TeachableMachine:
  """
Functions:
:meth:`~openpibo.vision.TeachableMachine.load`
:meth:`~openpibo.vision.TeachableMachine.predict`

  파이보의 카메라 Teachable Machine 기능을 사용합니다.

  * ``이미지 프로젝트`` 의 ``표준 이미지 모델`` 을 사용합니다.
  * ``Teachable Machine`` 에서 학습한 모델을 적용하여 추론할 수 있습니다.
  * 학습한 모델은 ``Tensorflow Lite`` 형태로 다운로드 해주세요.

  example::

    from openpibo.vision import TeachableMachine

    tm = TeachableMachine()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def load(self, model_path, label_path):
    """
    (내부 함수) Tflite 모델로 불러옵니다. (부동소수점/양자화) 모두 가능

    example::

      tm.load_tflite('model_unquant.tflite', 'labels.txt')

    :param str model_path: Teachable Machine의 모델파일

    :param str label_path: Teachable Machine의 라벨파일
    """

    with open(label_path, 'r') as f:
      c = f.readlines()
      class_names = [item.split(maxsplit=1)[1].strip('\n') for item in c]

    # Load TFLite model and allocate tensors
    self.interpreter = Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()

    # Get input and output tensors.
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()

    # check the type of the input tensor
    self.floating_model = self.input_details[0]['dtype'] == np.float32

    self.height = self.input_details[0]['shape'][1]
    self.width = self.input_details[0]['shape'][2]

    self.class_names = class_names

  def predict(self, img):
    """
    (내부 함수) Tflite 모델로 추론합니다.

    example::

      cm = Camera()
      img = cm.read()
      tm.predict(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 가장 높은 확률을 가진 클래스 명, 결과(raw 데이터)
    """

    try:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (self.width, self.height))
      image = Image.fromarray(img)

      # Add a batch dimension
      input_data = np.expand_dims(image, axis=0)

      if self.floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

      # feed data to input tensor and run the interpreter
      self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
      self.interpreter.invoke()

      # Obtain results and map them to the classes
      preds = self.interpreter.get_tensor(self.output_details[0]['index'])
      preds = np.squeeze(preds)
      return self.class_names[np.argmax(preds)], preds
    except Exception as ex:
      raise Exception('Teachable Machine Model did not load properly.')
