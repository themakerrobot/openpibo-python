# Device Simulator

파이보의 device를 제어하고 상태를 확인할 수 있는 툴 입니다.

인터넷 브라우저가 필요합니다. (`Chrome` 브라우저 사용 권장)

__사용 방법__

```bash
$ cd ~/openpibo-tools/device-simulator
$ sudo python3 main.py --port 8888
```

- 프로그램을 실행합니다.

  `--port` : 연결할 포트를 입력합니다. 만약 설정하지 않으면, 기본 포트는 `8888`입니다.

  이후 인터넷 브라우저에서 `http://<PIBO IP>:<PORT>`에 접속합니다.

    * 예: http://192.168.2.144:8888

  ![image-20210820112314329](images/device_simulator/devicesimulator.png)

  

- 좌측 입력 바를 조작하여 **Neopixel** 을 제어할 수 있습니다.

  ![image-20210907162156112](images/device_simulator/image-20210907162156112.png)

  ![image-20210907162233280](images/device_simulator/image-20210907162233280.png)

  

- 우측 테이블에서는 디바이스 상태 정보를 확인할 수 있습니다.

  - 배터리 잔량
  - 전원케이블 연결 상태
  - 전원버튼 누름상태
  - PIR센서 신호
  - 터치센서 신호
