# 서보모터 제어 프로그램

+ Command List
 - servo help
   + 명령어 사용법
 - servo version
   + 버전 정보
 - servo init
   + 초기화 (모터 컨트롤러에 설정된 초기화 값)
 - servo profile init
   + profile (offset, pos, record) 초기화
 - servo profile offset set 1 2 3 4 5 6 7 8 9 10
   + profile offset 설정 (offset - 모터 세부 조정을 위한 기준)
 - servo profile offset get
   + profile offset 조회
 - servo profile pos get
   + profile pos 조회 (모터 현재 위치)
 - servo profile record get
   + profile record 조회 (모터 이동 거리 누적)
 - servo mwrite pos0<-900 ~ 900> pos1<-900 ~ 900> ...
   + 10개 모터 한번에 이동
   + servo mwrite 모터 0의 각도*10 (-900 ~ 900) 모터 1의 각도 모터3의 각도 ... 모터 9의 각도 
 - servo move pos0<-900 ~ 900> pos1<-900 ~ 900> ... pos9<-900 ~ 900> time(ms)<0 ~ >
   + 10개 모터 한번에 이동 + 정해진 시간 내에 (정해진 시간에 따라 속도 가속도 자동 변경 - 정밀하지는 않음)
   + servo move 모터 0의 각도*10 (-900 ~ 900) 모터 1의 각도 모터3의 각도 ... 모터 9의 각도 시간 (ms 단위)
 - servo write channel<0 ~ 9> value<-900 ~ 900>
   + 1개 모터 이동
   + servo write 모터 번호 모터 각도
 - servo speed channel<0 ~ 9> value<0 ~ 255>
   + 1개 모터 속도 설정
   + servo speed 모터 번호 모터 속도 (0-255) 0은 속도 설정X (가장 빠름) 
 - servo speed all value1<0 ~ 255> ...
   + 10개 모터 속도 설정
   + servo speed  all 모터 0의 속도 모터 1의 속도 ... 모터 9의 속도
 - servo accelerate channel<0 ~ 9> value<0 ~ 255>
   + 1개 모터 가속도 설정
   + servo accelerate 모터 번호 모터 가속도 (0-255) 0은 가속도 설정X (등속)
 - servo accelerate all value1<0 ~ 255> ...
   + 10개 모터 가속도 설정
   + servo accelerate all 모터 0의 가속도 모터 1의 가속도 ... 모터 9의 가속도

 - For Example:
  * servo write 0 0
