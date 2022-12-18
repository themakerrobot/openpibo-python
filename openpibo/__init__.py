"""
openpibo-python
"""
import os, sys, json, shutil

__version__ = '0.9.2.43'

defconfig = {"kakaokey":"","datapath":"/home/pi/openpibo-files", "napi_host":"https://oe-napi.circul.us", "sapi_host":"https://oe-sapi.circul.us", "robotId":"", "eye":"0,0,0,0,0,0"}
config_path = '/home/pi/config.json'

try:
  if os.path.isfile(config_path) == False:
    config = defconfig
  else:
    with open(config_path, 'r') as f:
      config = json.load(f)
    for k, v in defconfig.items():
      if k not in config:
        config[k] = v
except Exception as ex:
  config = defconfig
finally:
  with open(config_path, 'w') as f:
    json.dump(config, f)

for k, v in config.items():
  exec('{}="{}"'.format(k,v))

shutil.chown(config_path, 'pi', 'pi')
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
