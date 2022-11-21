"""
openpibo-python
"""
import os, sys, json, shutil

__version__ = '0.9.2.38'

if os.path.isfile('/home/pi/config.json') == False:
  config = {"datapath":"/home/pi/openpibo-files", "napi_host":"", "sapi_host":"", "robotId":"", "eye":"0,0,0,0,0,0"}
  with open('/home/pi/config.json', 'w') as f:
    json.dump(config, f)
else:
  with open('/home/pi/config.json', 'r') as f:
    config = json.load(f)

for k, v in config.items():
  exec('{}="{}"'.format(k,v))

shutil.chown('/home/pi/config.json', 'pi', 'pi')
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from .modules import *

'''
from .audio import Audio
from .collect import Wikipedia, Weather, News
from .device import Device
from .motion import Motion
from .oled import Oled
from .speech import Speech, Dialog
from .vision import Camera, Face, Detect
from .edu_v1 import Pibo
'''
