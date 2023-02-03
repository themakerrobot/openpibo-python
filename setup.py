from setuptools import setup, find_packages
from openpibo import __version__ as VERSION

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
        'opencv-contrib-python>=4.5.4.60',
        'dlib>=19.24.0',
        'pyzbar>=0.1.8',
        'beautifulsoup4>=4.6.0',
        'konlpy>=0.5.2',
        'tweepy>=3.10.0',
        'future>=0.18.2',
        'pillow>=7.2.0',
        'RPi.gpio>=0.7.0',
        'pyserial>=3.5',
        'requests>=2.28.1',
        'tflite-runtime>=2.5.0',
        'openpibo_models>=0.4.5',
        'openpibo_face_models>=0.4.3',
        'openpibo_detect_models>=0.4.4',
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
