"""
`OpenCV` 라이브러리를 활용한 PIBO의 영상처리 관련 라이브러리입니다.

카메라 기능, 얼굴 인식, 객체/바코드/문자 인식을 수행합니다.

Class:
:obj:`~openpibo.vision.Camera`
:obj:`~openpibo.vision.Face`
:obj:`~openpibo.vision.Detect`
:obj:`~openpibo.vision.TeachableMachine`
"""

import cv2,dlib
import os,pickle
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

try:
  from tensorflow import keras
except Exception as ex:
  print("Warn:", ex)

class Camera:
  """
Functions:
:meth:`~openpibo.vision.Camera.imread`
:meth:`~openpibo.vision.Camera.read`
:meth:`~openpibo.vision.Camera.resize`
:meth:`~openpibo.vision.Camera.imwrite`
:meth:`~openpibo.vision.Camera.rectangle`
:meth:`~openpibo.vision.Camera.circle`
:meth:`~openpibo.vision.Camera.putTextPIL`
:meth:`~openpibo.vision.Camera.putText`
:meth:`~openpibo.vision.Camera.cartoonize`
:meth:`~openpibo.vision.Camera.convert_img`
:meth:`~openpibo.vision.Camera.rotate`
:meth:`~openpibo.vision.Camera.bgrhls`

  파이보의 카메라를 제어합니다.

  * 사진 촬영, 읽기, 쓰기 등 카메라 기본 기능을 사용할 수 있습니다.
  * Cartoonize 기능을 사용할 수 있습니다.

  example::

    from openpibo.vision import Camera

    pibo_camera = Camera()
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

      pibo_camera.imread('/home/pi/openpibo-files/data/image/clear.png')

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

      pibo_camera.read()

    :returns: ``numpy.ndarray`` 타입 이미지 객체
    """

    # for opencv buffering issue
    self.cap.grab()
    self.cap.grab()
    return self.cap.read()[1]

  def resize (self, img, w, h):
    """
    Opencv 이미지의 크기를 변환합니다.

    example::

      img = pibo_camera.read()
      pibo_camera.resize(img, 128, 64)

    :param numpy.ndarray img: 이미지 객체

    :param int w: 변환될 이미지의 가로 크기입니다. (픽셀 단위)

    :param int h: 변환될 이미지의 세로 크기입니다. (픽셀 단위)

    :returns: 크기 변환 후의 이미지 객체
    """

    return cv2.resize(img, (w, h))

  def imwrite(self, filename, img):
    """
    이미지를 파일로 저장합니다.

    example::

      img = pibo_camera.read()
      pibo_camera.imwrite('/home/pi/image.jpg', img)

    :param str filename: 저장할 파일 경로

      확장자는 jpg 또는 png를 사용할 수 있습니다.

    :param numpy.ndarray img: 저장할 이미지 객체
    """

    return cv2.imwrite(filename, img)

  def rectangle(self, img, p1, p2, colors=(255,255,255), tickness=1):
    """
    이미지에 직사각형을 그립니다.

    example::

      img = pibo_camera.read()
      pibo_camera.rectangle(img, (10, 10), (300, 200), (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param tuple(int, int) p1: 좌측상단 좌표 (x, y)

    :param tuple(int, int) p2: 우측하단 좌표 (x, y)

    :param tuple(int, int, int) colors: RGB 값 (r, g, b)

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

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.rectangle(img, p1, p2, colors, tickness)

  def circle(self, img, p, r, colors=(255,255,255), tickness=1):
    """
    이미지에 직사각형을 그립니다.

    example::

      img = pibo_camera.read()
      pibo_camera.circle(img, (10, 10), (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param tuple(int, int) p: 좌측상단 좌표 (x, y)

    :param int r: 반지름

    :param tuple(int, int, int) colors: RGB 값 (r, g, b)

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

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.circle(img, p, r, colors, tickness)

  def putTextPIL(self, img, text, points, size=30, colors=(255,255,255)):
    """
    이미지에 문자를 입력합니다. (한/영 가능 - pillow 이용)

    example::

      img = pibo_camera.read()
      new_img = pibo_camera.putText(img, '안녕하세요.', (15, 10), 30, (255, 255, 255))

    :param numpy.ndarray img: 이미지 객체

    :param str text: 표시할 문자열

    :param tuple(int, int) points: 텍스트 블록 좌측상단 좌표 (x, y)

    :param int size: 표시할 글자의 크기

    :param tuple(int, int, int) colors: 글자 색깔 RGB 값 (b, g, r)

    """
    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 2:
      raise Exception(f'len({points}) must be 2')

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

      img = pibo_camera.read()
      new_img = pibo_camera.putText(img, 'hello', (15, 10), 10, (255, 255, 255), 1)

    :param numpy.ndarray img: 이미지 객체

    :param str text: 표시할 문자열

    :param tuple(int, int) points: 텍스트 블록 좌측하단 좌표 (x, y)

    :param int size: 표시할 글자의 크기

    :param tuple(int, int, int) colors: 글자 색깔 RGB 값 (b, g, r)

    :param int tickness: 글자 두께
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if type(points) is not tuple:
      raise Exception(f'"{points}" must be tuple type')

    if len(points) != 2:
      raise Exception(f'len({points}) must be 2')

    if type(colors) is not tuple:
      raise Exception(f'"{colors}" must be tuple type')

    if len(colors) != 3:
      raise Exception(f'len({colors}) must be 3')

    return cv2.putText(img, text, points, cv2.FONT_HERSHEY_SIMPLEX, size, colors, tickness)

  def cartoonize(self, img):
    """
    만화같은 이미지로 변경합니다.

    example::

      img = pibo_camera.read()
      new_image = pibo_camera.cartoonize(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 변환된 ``numpy.ndarray`` 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    numDownSamples = 2 # number of downscaling steps
    numBilateralFilters = 7  # number of bilateral filtering steps

    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img
    for _ in range(numDownSamples):
      img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
      img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    # upsample image to original size
    for _ in range(numDownSamples):
      img_color = cv2.pyrUp(img_color)

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed
    # with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)

  def convert_img(self, img, w=128, h=64):
    """
    이미지의 크기를 변환합니다.

    example::

      img = pibo_camera.read()
      pibo_camera.convert_img(img, 128, 64)

    :param numpy.ndarray img: 이미지 객체

    :param int w: 변환될 이미지의 가로 크기입니다. (픽셀 단위)

    :param int h: 변환될 이미지의 세로 크기입니다. (픽셀 단위)

    :returns: 크기 변환 후의 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2GRAY)

  def rotate(self, img, degree=10, ratio=0.9):
    """
    이미지를 회전시킵니다.

    example::

      img = pibo_camera.read()
      pibo_camera.rotate(img, 10, 0.9)

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

  def bgr_hls(self, img):
    """
    BGR 이미지 모델을 HLS 이미지 모델로 변환한다.

    BGR: Blue, Green, Red

    HLS: Hue(색상), Luminance(명도), Saturation(채도)

    example::

      img = pibo_camera.read()
      new_img = pibo_camera.bgr_hls(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 변환된 ``numpy.ndarray`` 이미지 객체
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

class Face:
  """
Functions:
:meth:`~openpibo.vision.Face.init_db`
:meth:`~openpibo.vision.Face.detect`
:meth:`~openpibo.vision.Face.train_face`
:meth:`~openpibo.vision.Face.recognize`
:meth:`~openpibo.vision.Face.get_ageGender`
:meth:`~openpibo.vision.Face.get_db`
:meth:`~openpibo.vision.Face.save_db`
:meth:`~openpibo.vision.Face.delete_face`
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

    pibo_face = Face()
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
    self.face_detector = cv2.CascadeClassifier(openpibo_face_models.filepath("haarcascade_frontalface_default.xml"))
    self.predictor = dlib.shape_predictor(openpibo_dlib_models.filepath("shape_predictor_5_face_landmarks.dat"))
    self.face_encoder = dlib.face_recognition_model_v1(openpibo_dlib_models.filepath("dlib_face_recognition_resnet_model_v1.dat"))

    self.emotion_class_names = ['angry','disgusted','fearful','happy','sad','surprised','neutral']
    self.emotion_interpreter = Interpreter(model_path=openpibo_face_models.filepath("emotion.tflite"))
    self.emotion_interpreter.allocate_tensors()
    self.emotion_input_details = self.emotion_interpreter.get_input_details()
    self.emotion_output_details = self.emotion_interpreter.get_output_details()

  def detect(self, img):
    """
    얼굴을 탐색합니다.

    example::

      img = pibo_camera.read()
      pibo_face.detect(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 인식된 얼굴들의 (x, y, w, h) 배열 입니다.

      list 타입으로, 이미지 하나에 얼굴이 여러 개 인식된 경우 인식된 얼굴의 좌표가 모두 입력됩니다.

      example::

        [(10, 10, 40, 50), (120, 30, 160, 70), (130, 140, 200, 260)]
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return self.face_detector.detectMultiScale(gray, 1.1, 5) # [(x,y,w,h), ...]

  def get_ageGender(self, img, face):
    """
    얼굴의 나이, 성별을 추정합니다. (deprecate soon)

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      face = faces[0] # face는 faces중 하나
      pibo_face.get_ageGender(img, face)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray face: 얼굴의 좌표 (x, y, w, h)

    :returns: ``{"age": 나이, "gender": 성별}``

      * age: 나이의 범위를 tuple() 형태로 출력한다.

        ex) (15, 20) # 15살에서 20살 정도

      * gender: ``male`` / ``female``

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    if len(face) != 4:
      raise Exception('"face" must be [x,y,w,h]')

    x, y, w, h = face
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

    # visualize
    #cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
    #cv2.putText(img, "{} {}".format(gender, age), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,128), 2)
    return {"age":age, "gender":gender}

  def get_age(self, face):
    """
    얼굴의 나이를 추정합니다.

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      x,y,w,h = faces[0] # face는 faces중 하나
      pibo_face.get_age(img[y:y+h, x:x+w])

    :param numpy.ndarray face: 얼굴 이미지

    :returns: ``나이 범위`` ``Raw데이터``

      * age: 나이의 범위를 tuple() 형태로 출력한다.

        ex) (15, 20) # 15살에서 20살 정도

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(face) is np.ndarray:
      raise Exception('"face" must be image data from opencv')

    face_img = face.copy()
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict age
    self.agenet.setInput(blob)
    age_preds = self.agenet.forward()
    age = self.age_class[age_preds[0].argmax()]

    return age, age_preds[0]

  def get_gender(self, face):
    """
    얼굴의 성별을 추정합니다.

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      x,y,w,h = faces[0] # face는 faces중 하나
      pibo_face.get_gender(img[y:y+h, x:x+w])

    :param numpy.ndarray face: 얼굴 이미지

    :returns: ``성별`` ``Raw데이터``

      * gender: ``male`` / ``female``

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(face) is np.ndarray:
      raise Exception('"face" must be image data from opencv')

    face_img = face.copy()
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict gender
    self.gendernet.setInput(blob)
    gender_preds = self.gendernet.forward()
    gender = self.gender_class[gender_preds[0].argmax()]

    return gender, gender_preds[0]

  def get_emotion(self, face):
    """
    얼굴의 감정을 추정합니다.

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      x,y,w,h = faces[0] # face는 faces중 하나
      pibo_face.get_gender(img[y:y+h, x:x+w])

    :param numpy.ndarray face: 얼굴 이미지

    :returns: ``감정`` ``Raw데이터``

      * emotion: ``angry``, ``disgusted``, ``fearful``, ``happy``, ``sad``, ``surprised``, ``neutral``

    참고: https://github.com/kairess/age_gender_estimation
    """

    if not type(face) is np.ndarray:
      raise Exception('"face" must be image data from opencv')

    img = face.copy()

    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 96))
    image = Image.fromarray(img)

    # Add a batch dimension
    input_data = np.expand_dims(image, axis=0).reshape(-1, 96, 96, 1)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # feed data to input tensor and run the interpreter
    self.emotion_interpreter.set_tensor(self.emotion_input_details[0]['index'], input_data)
    self.emotion_interpreter.invoke()

    # Obtain results and map them to the classes
    preds = self.emotion_interpreter.get_tensor(self.emotion_output_details[0]['index'])
    preds = np.squeeze(preds)
    return self.emotion_class_names[np.argmax(preds)], preds.tolist()

  def init_db(self):
    """
    얼굴 데이터베이스를 초기화합니다.

    초기화된 데이터베이스는 빈 이중 list ``[[], []]`` 입니다.

    example::

      pibo_face.init_db()
    """

    self.facedb = [[], []]

  def train_face(self, img, face, name):
    """
    얼굴을 학습하여 얼굴 데이터베이스에 저장합니다.

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      face = faces[0] # face는 faces중 하나
      pibo_face.train_face(img, face, 'honggildong')

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray face: 디텍팅한 얼굴의 사각형 좌측상단, 우측하단 포인트 (x1, y1, x2, y2)

    :param str name: 디텍팅한 얼굴에 붙일 이름
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    if len(face) != 4:
      raise Exception('"face" must be [x,y,w,h]')

    x,y,w,h = face
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

      pibo_face.delete_face('honggildong')

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

  def recognize(self, img, face):
    """
    등록된 얼굴을 인식합니다.

    example::

      img = pibo_camera.read()
      faces = pibo_face.detect(img)
      face = faces[0] # face는 faces중 하나
      pibo_face.recognize(img, face)

    :param numpy.ndarray img: 이미지 객체

    :param numpy.ndarray face: 얼굴의 좌표 (x, y, w, h)

    :returns: ``{"name": 이름, "score": 오차도}``

      얼굴이 비슷할수록 오차도가 낮게 측정됩니다.

      오차도가 0.4 이하일 때 동일인으로 판정합니다.
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv') 

    if len(face) != 4:
      raise Exception('"face" must be [x,y,w,h]')

    if len(self.facedb[0]) < 1:
      return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data={"name":"Guest", "score":0}
    (x,y,w,h) = face
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
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

      pibo_face.get_db()

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

      pibo_face.save_db('/home/pi/facedb')

    :param str filename: 저장할 얼굴 데이터베이스 파일의 경로입니다.
    """

    with open(filename, "w+b") as f:
      pickle.dump(self.facedb, f)

  def load_db(self, filename):
    """
    얼굴 데이터베이스 파일을 불러옵니다.

    example::

      pibo_face.load_db('/home/pi/facedb')

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

  인식과 관련된 다양한 기능을 사용할 수 있는 클래스입니다.

  * 90개 class 안에서의 객체 인식 (MobileNet V2)
  * QR/바코드 인식 (pyzbar)
  * Pose 인식

  example::

    from openpibo.vision import Detect

    pibo_detect = Detect()
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

      img = pibo_camera.read()
      pibo_detect.detect_object(img)

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

      img = pibo_camera.read()
      pibo_detect.detect_qr(img)

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

      img = pibo_camera.read()
      pibo_detect.detect_pose(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: ``{"data": 인식한 결과, "img": Pose를 반영한 이미지 }``
    """

    if not type(img) is np.ndarray:
      raise Exception('"img" must be image data from opencv')

    list_persons = [self.pose_detector.detect(img)]
    img = visualize(img, list_persons)
    return {"data":list_persons, "img":img}

class TeachableMachine:
  """
Functions:
:meth:`~openpibo.vision.TeachableMachine.load_tflite`
:meth:`~openpibo.vision.TeachableMachine.predict_tflite`
:meth:`~openpibo.vision.TeachableMachine.load_keras`
:meth:`~openpibo.vision.TeachableMachine.predict_keras`
:meth:`~openpibo.vision.TeachableMachine.load`
:meth:`~openpibo.vision.TeachableMachine.predict`


  파이보의 카메라 Teachable Machine 기능을 사용합니다.

  * ``이미지 프로젝트`` 의 ``표준 이미지 모델`` 을 사용합니다.
  * ``Teachable Machine`` 에서 학습한 모델을 적용하여 추론할 수 있습니다.
  * 학습한 모델은 ``Tensorflow Lite`` 또는 ``Keras`` 형태로 다운로드 해주세요.

  example::

    from openpibo.vision import TeachableMachine

    tm = TeachableMachine()
    # 아래의 모든 예제 이전에 위 코드를 먼저 사용합니다.
  """

  def __init__(self):
    self.mode = None

  def load_tflite(self, model_path, label_path):
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

  def predict_tflite(self, img):
    """
    (내부 함수) Tflite 모델로 추론합니다.

    example::

      cm = Camera()
      img = cm.read()
      tm.predict(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 가장 높은 확률을 가진 클래스 명, 결과(raw 데이터)
    """

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

  def load_keras(self, model_path, label_path):
    """
    (내부 함수) Keras 모델을 불러옵니다.

    example::

      tm.load('keras_model.h5', 'labels.txt')

    :param str model_path: Teachable Machine의 모델파일

    :param str label_path: Teachable Machine의 라벨파일
    """

    with open(label_path) as f:
      c = f.read().split('\n')
      class_names = [c[i].split(maxsplit=1)[1] for i in range(len(c)-1)]

    self.model = keras.models.load_model(model_path, compile=False)
    self.class_names = class_names

  def predict_keras(self, img):
    """
    (내부 함수) Keras 모델로 추론합니다.

    example::

      cm = Camera()
      img = cm.read()
      tm.predict(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 가장 높은 확률을 가진 클래스 명, 결과(raw 데이터)
    """

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1
    predictions = self.model.predict(img)[0]
    return self.class_names[np.argmax(predictions)], predictions

  def load(self, model_path, label_path):
    """
    Teachable Machine (Tflite, Keras) 모델을 초기화 합니다.

    example::

      tm.load('keras_model.h5', 'labels.txt')

    :param str model_path: Teachable Machine의 모델파일

    :param str label_path: Teachable Machine의 라벨파일
    """

    if model_path.rsplit('.')[-1] == 'tflite':
      self.load_tflite(model_path, label_path)
      self.mode = 'tflite'
    elif model_path.rsplit('.')[-1] == 'h5':
      self.load_keras(model_path, label_path)
      self.mode = 'keras'
    else:
      raise Exception('can load Teachable Machine model "tflite" or "h5"')

  def predict(self, img):
    """
    적용한 Teachable Machine (Tflite, Keras) 모델을 기반으로 추론합니다.

    example::

      cm = Camera()
      img = cm.read()
      tm.predict(img)

    :param numpy.ndarray img: 이미지 객체

    :returns: 가장 높은 확률을 가진 클래스 명, 결과(raw 데이터)
    """

    if self.mode == 'tflite':
      return self.predict_tflite(img)
    elif self.mode == 'keras':
      return self.predict_keras(img)
    else:
      raise Exception('did not load Teachable Machine Model.')
