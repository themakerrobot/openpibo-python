"""
교육용 파이보 패키지
"""
import os, sys, json

__version__ = '0.8.2'

with open('/home/pi/config.json', 'r') as f:
  config = json.load(f)

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from .modules import *

from .audio import Audio
from .collect import Wikipedia, Weather, News
from .device import Device
from .motion import Motion, PyMotion
from .oled import Oled
from .speech import Speech, Dialog
from .vision import Camera, Face, Detect
from .edu_v1 import Pibo
