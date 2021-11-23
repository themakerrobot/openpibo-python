from setuptools import setup, find_packages
from openpibo import __version__ as VERSION

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
        'opencv-contrib-python>=4.1.0.25',
        'dlib>=19.19.0',
        'pyzbar==0.1.8',
        'pytesseract==0.3.4',
        'beautifulsoup4==4.6.0',
        'konlpy==0.5.2',
        'future==0.18.2',
        'pillow==7.2.0',
        'RPi.gpio==0.7.0',
        'pyserial==3.5',
        'requests==2.25.1',
        'pytest==6.2.4',
        'rich==10.6.0',
        'flask==2.0.1',
        'flask-socketio==5.1.1',
        'openpibo_models>=0.4.1',
        'openpibo_face_models>=0.4.1',
        'openpibo_detect_models>=0.4.1',
        'openpibo_dlib_models>=0.4.1',
]

test_requirements = [
]

setup(
    name                = 'openpibo-python',
    version             = VERSION,
    description         = 'openpibo-python package.',
    long_description    = readme,
    author              = "circulus",
    author_email        = 'leeyunjai@circul.us',
    url                 = 'https://github.com/themakerrobot/openpibo-python',
    packages            = find_packages(),
    install_requires    = requirements,
    keywords            = 'openpibo',
    classifiers         = [
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite          = 'tests',
    tests_require       = test_requirements
)
