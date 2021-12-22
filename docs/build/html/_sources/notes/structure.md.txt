# 라이브러리 구조

openpibo / openpibo_models 패키지 구조에 관해 설명합니다.

openpibo 패키지는 파이보를 제어하기 위한 내부 modules과 libraries 부분으로 나뉘어 있습니다.
openpibo_models 패키지는 openpibo 패키지를 사용하기 위한 폰트/모션 파일, 인공지능 모델 등의 파일을 가지고 있습니다.

## openpibo/modules

라이브러리를 사용하는 데 필요한 각종 모듈입니다.

사용되는 라이브러리의 종류에 따라 분류되어있습니다.

```
modules
├── __init__.py
├── oled
│   ├── __init__.py
│   ├── board.py
│   ├── busio.py
│   ├── chip.py
│   ├── digitalio.py
│   ├── framebuf.py
│   ├── pure_spi.py
│   ├── spi.py
│   ├── spi_device.py
│   ├── ssd1306.py
│   └── util.py
└── speech
    ├── __init__.py
    ├── constant.py
    └── google_trans_new.py
```

## openpibo/libraries

파이보의 다양한 기능을 사용할 수 있는 Class가 저장된 파일입니다.

```
openpibo
├── audio.py
├── collect.py
├── device.py
├── motion.py
├── oled.py
├── speech.py
├── vision.py
└── edu_v1.py
```

세부 가이드는 아래의 링크를 참조해주세요.

- [audio.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/audio.html)
- [collect.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/collect.html)
- [device.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/device.html)
- [motion.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/motion.html)
- [oled.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/oled.html)
- [speech.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/speech.html)
- [vision.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/vision.html)
- [edu_v1.py](https://themakerrobot.github.io/openpibo-python/build/html/libraries/edu_v1.html)


## openpibo-models
폰트 파일, 모션 데이터베이스, 그리고 각종 인공지능 모델 등 라이브러리를 사용하기 위한 최소한의 데이터가 들어있습니다.

```
openpibo-models/
├── openpibo_models
|   ├── KDL.ttf
|   ├── motion_db.json
|   ├── sample_db.json
|   └── dialog.csv
├── openpibo_detect_models
|   ├── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
|   └── frozen_inference_graph.pb
├── openpibo_face_models
|   ├── deploy_age.prototxt
|   ├── age_net.caffemodel
|   ├── deploy_gender.prototxt
|   ├── gender_net.caffemodel
|   └── haarcascade_frontalface_default.xml
└── openpibo_dlib_models
    ├── dlib_face_recognition_resnet_model_v1.dat
    └── shape_predictor_5_face_landmarks.dat
```

- **폰트**

   - **KDL.ttf**

      OLED에 텍스트를 출력할 때 사용하는 기본 글씨체입니다.

      ![](images/structure/kdl.jpg)

- **모션 데이터베이스**

   - **motion_db.json**
   
      파이보에 기본적인 동작이 저장되어있는 데이터베이스로, 저장된 모션 리스트는 다음과 같습니다.

      ```
      stop, stop_body, sleep, lookup, left, left_half, right, right_half, foward1-2,
      backward1-2, step1-2, hifive, cheer1-3, wave1-6, think1-4, wake_up1-3, hey1-2,
      yes_h, no_h, breath1-3, breath_long, head_h, spin_h, clapping1-2, hankshaking,
      bow, greeting, hand1-4, foot1-2, speak1-2, speak_n1-2, speak_q, speak_r1-2, 
      speak_l1-2, welcome, happy1-3, excite1-2, boring1-2, sad1-3, handup_r, 
      handup_l, look_r, look_l, dance1-5, motion_test, test1-4
      # foward1-2는 forward1, forward2 두 종류가 있음을 의미합니다.
      ```

   - **sample_db.json**
   
      모션 데이터베이스의 양식이 저장되어있습니다.

      ```json
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
      ```

   새로운 모션을 만들기 위해서는 sample_db.json의 양식을 갖춰야 하며, [Motion Creator](https://themakerrobot.github.io/openpibo-python/build/html/tools/motion_creator.html)를 이용해 간편히 만들 수 있습니다.

- **대화 데이터셋**

   - **dialog.csv**

      질문과 답변 형식의 대화 데이터셋입니다.

      챗봇 기능을 사용할 때 질문에 대한 대답을 추론하기 위해 사용됩니다.
   
- **인공지능 모델**

   vision 라이브러리에서 사용되는 각종 인공지능 모델입니다.

   - **frozen_inference_graph.pb, ssd_mobilenet_v2_coco_2018_03_29.pbtxt**

      객체 인식을 사용하기 위한 인공지능 네트워크 구조와 모델입니다.
   
   - **deploy_age.prototxt, age_net.caffemodel**

      인물의 나이를 가늠하기 위한 인공지능 네트워크 구조와 모델입니다.
   
   - **deploy_gender.prototxt, gender_net.caffemodel**

      인물의 성별을 추론하기 위한 인공지능 네트워크 구조와 모델입니다.
   
   아래는 얼굴인식을 위한 모델입니다.

   - **dlib_face_recognition_resnet_model_v1.dat** # face_encoder

      얼굴을 인식하여 Numpy 배열로 변환하는 인공지능 모델입니다.

   - **haarcascade_frontalface_default.xml** # face_detector

      얼굴을 검출하도록 미리 학습시켜 놓은 XML 포맷으로 저장된 검출기입니다.

   - **shape_predictor_5_face_landmarks.dat** # predictor

      얼굴에 5개의 특징점을 추출하여 표정을 예측하는 인공지능 모델입니다.

## 참고 라이브러리 소개
- **NLP(자연어처리)**

   - **MeCab**

     일본에서 만든 형태소 분석 엔진

     언어, 사전 코퍼스에 의존하지 않는 범용적인 설계

     품사 독립적 설계

     각종 스크립트 언어 바인딩 (perl / ruby / python / java / C#)


- **Computer Vision(영상처리)**

   - **OpenCV (Open source Computer Vision)**

     영상 처리 및 컴퓨터 비전 관련 오픈소스

     이미지, 영상처리, Object Detection, Motion Detection 등의 기능을 제공합니다.

   - **Tensorflow**

      머신러닝/딥러닝 프레임워크

      구글 내 연구와 제품개발을 위한 목적으로 구글 브레인팀이 만들었고, 오픈 소스로 공개하였습니다.

      pytorch와 함께 가장 많이 쓰는 프레임워크입니다.

   - **Dlib**

      이미지 처리 및 기계 학습, 얼굴 인식 등을 할 수 있는 C++로 개발된 고성능의 라이브러리
      
      facial landmarks를 통해 얼굴을 검출하는 기능이 많이 사용됩니다. (파이보에서는 5개의 face_landmarks를 찾습니다.)

   - **Tesseract**

      다양한 운영체제를 지원하기 위한 OCR(Optical Character Recognition) 엔진

      OCR 이미지로부터 텍스트를 인식하고, 추출합니다.
   
      오프라인 문자인식 기법으로 입력된 input 이미지의 특징점을 추출하고 그 특징점을 사용하여 문자를 인식합니다.

   - **Pyzbar**

     비디오 스트림, 이미지 파일 및 이미지와 같은 다양한 소스에서 바코드를 판독할 수 있는 오픈소스 라이브러리

   - **Numpy**

     벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
     
     array(행렬) 단위로 데이터를 관리하며 이에 대해 연산을 수행합니다.

