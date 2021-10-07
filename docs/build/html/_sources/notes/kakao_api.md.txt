# KAKAO API

본 문서는 `KAKAO REST API KEY` 를 발급받는 방법에 대해 안내합니다.

**Speech** 라이브러리의 음성인식(**STT**), 음성합성(**TTS**) 기능은 KAKAO API를 사용하기 때문에 `KAKAO REST API KEY` 가 필요합니다.
키 발급 방법은 [kakao developers](https://developers.kakao.com/)에 회원가입 후 다음 과정으로 진행합니다.

1. 로그인 후 [내 애플리케이션] 클릭

   ![api1](images/kakao_api/api1.png)

2. [애플리케이션 추가하기] 클릭

   ![api2](images/kakao_api/api2.png)

3. 앱 이름 및 사업자명 입력 후 저장

   ![api3](images/kakao_api/api3.png)

4. 새로 생성한 애플리케이션 클릭

   ![api4](images/kakao_api/api4.png)

5. config.py에 발급받은 REST API 키 입력 후, 왼쪽의 [음성] 클릭

   ![api5](images/kakao_api/api5.png)

6. 이후 `/home/pi/config.json`의 `KAKAO_ACCOUNT`에 발급받은 `REST API 키` 입력

   ```json
   {
       "DATA_PATH":"/home/pi/openpibo-files/data",
       "KAKAO_ACCOUNT": "<여기에 발급받은 REST API 키를 입력해주세요>",
       "robotId": ""
   }
   ```

7. 활성화 설정의 [OFF] 버튼 클릭

   ![api6](images/kakao_api/api6.png)

8. 사용 목적 입력 후 저장

   ![api7](images/kakao_api/api7.png)

9. 활성화 설정의 상태가 [ON]으로 바뀌면 완료

   ![api8](images/kakao_api/api8.png)


