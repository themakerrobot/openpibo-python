current_type = {"O":"예", "X":"아니오", "+":"더하기", "-":"빼기", "*":"곱하기", "%":"나누기"}
composer_type = {"DO":"도", "RE":"레", "MI":"미", "FA":"파", "SOL":"솔", "LA":"라", "TI":"시", "DO2":"도2"}
fucntion_type = {"WEATHER":"날씨", "NEWS":"뉴스", "DANCE":"댄스", "MUSIC":"음악", "EXERCISE":"체조", "CAMERA":"카메라"}
action_type = {"RUN":"수행", "REPEAT":"반복", "DIALOG":"대화"}
eye_type = {"255,0,0":"빨강", "255,128,0":"주황", "255,255,0":"노랑", "0,255,0":"초록", "0,0,255":"파랑", "255,20,147":"핑크", "128,0,128":"보라"}
motion_type = {"RIGHT":"우회전", "LEFT":"좌회전", "FORWARD":"전진", "BACKWARD":"후진", "BORING":"지루함", "EXCITE":"신남", "SAD":"슬픔", "HAPPY":"기쁨"}

def get_card(data):
  def decodeQR(code):
    orgData = ''
    for item in code:
      if ord(item) <= 10000:
        orgData += chr(ord(item)-1)
      else:
        orgData += chr(ord(item)+1)
    return orgData;  

  data = decodeQR(data)
  arr = data.split("|")
  res = None

  if arr[0] == '!':
    if arr[1] == '_CURRENT':
      d = arr[len(arr)-1].replace('"','')
      if d in ["O", "X", "+", "-", "*", "%"]:
        res = current_type[d]
      elif d in ["0","1","2","3","4","5","6","7","8","9"]:
        res = d
      else:
        pass
    elif 'OFFICIAL' in arr[1]:
      d = arr[len(arr)-1].replace('"','')
      if arr[1] == 'OFFICIAL_GAME':
        res = d.split("_")[2]
      elif arr[1] == 'OFFICIAL_COMPOSER':
        res = composer_type[d]
      elif arr[1].split("_")[1] in fucntion_type:
        res = fucntion_type[arr[1].split('_')[1]]
      else:
        pass
    elif arr[1] == '_PIBO':
      d = arr[len(arr)-1].replace('"','')
      if arr[2] == '$_ACTION':
        res = action_type[d]
      elif arr[2] == '$_EYE':
        res = eye_type[d]
      elif arr[2] == '$_MOTION':
        res = motion_type[d]
      else:
        pass
    else:
      pass
    return res

if __name__ == "__main__":

  test_data = {
    '!|_CURRENT|$_BC|"0"': '0',
    '!|_CURRENT|$_BC|"1"': '1',
    '!|_CURRENT|$_BC|"2"': '2',
    '!|_CURRENT|$_BC|"3"': '3',
    '!|_CURRENT|$_BC|"4"': '4',
    '!|_CURRENT|$_BC|"5"': '5',
    '!|_CURRENT|$_BC|"6"': '6',
    '!|_CURRENT|$_BC|"7"': '7',
    '!|_CURRENT|$_BC|"8"': '8',
    '!|_CURRENT|$_BC|"9"': '9',

    '!|_CURRENT|$_BC|"O"': '예',
    '!|_CURRENT|$_BC|"X"': '아니오',
    '!|_CURRENT|$_BC|"+"': '더하기',
    '!|_CURRENT|$_BC|"-"': '빼기',
    '!|_CURRENT|$_BC|"%"': '나누기',
    '!|_CURRENT|$_BC|"*"': '곱하기',

    '!|OFFICIAL_GAME|$_BC|"E_A_ALLIGATOR"': 'ALLIGATOR',
    '!|OFFICIAL_GAME|$_BC|"E_B_BEAR"': 'BEAR',
    '!|OFFICIAL_GAME|$_BC|"E_C_CAT"': 'CAT',
    '!|OFFICIAL_GAME|$_BC|"E_D_DOG"': 'DOG',
    '!|OFFICIAL_GAME|$_BC|"E_E_ELEPHANT"': 'ELEPHANT',
    '!|OFFICIAL_GAME|$_BC|"E_F_FROG"': 'FROG',
    '!|OFFICIAL_GAME|$_BC|"E_G_GOAT"': 'GOAT',
    '!|OFFICIAL_GAME|$_BC|"E_H_HORSE"': 'HORSE',
    '!|OFFICIAL_GAME|$_BC|"E_I_IMPALA"': 'IMPALA',
    '!|OFFICIAL_GAME|$_BC|"E_J_JAGUAR"': 'JAGUAR',
    '!|OFFICIAL_GAME|$_BC|"E_K_KANGAROO"': 'KANGAROO',
    '!|OFFICIAL_GAME|$_BC|"E_L_LION"': 'LION',
    '!|OFFICIAL_GAME|$_BC|"E_M_MONKEY"': 'MONKEY',
    '!|OFFICIAL_GAME|$_BC|"E_N_NEWT"': 'NEWT',
    '!|OFFICIAL_GAME|$_BC|"E_O_OWL"': 'OWL',
    '!|OFFICIAL_GAME|$_BC|"E_P_PIG"': 'PIG',
    '!|OFFICIAL_GAME|$_BC|"E_Q_QUAIL"': 'QUAIL',
    '!|OFFICIAL_GAME|$_BC|"E_R_RACCOON"': 'RACCOON',
    '!|OFFICIAL_GAME|$_BC|"E_S_SHEEP"': 'SHEEP',
    '!|OFFICIAL_GAME|$_BC|"E_T_TIGER"': 'TIGER',
    '!|OFFICIAL_GAME|$_BC|"E_U_UNICORN"': 'UNICORN',
    '!|OFFICIAL_GAME|$_BC|"E_V_VULTURE"': 'VULTURE',
    '!|OFFICIAL_GAME|$_BC|"E_W_WOLF"': 'WOLF',
    '!|OFFICIAL_GAME|$_BC|"E_X_XERUS"': 'XERUS',
    '!|OFFICIAL_GAME|$_BC|"E_Y_YAK"': 'YAK',
    '!|OFFICIAL_GAME|$_BC|"E_Z_ZEBRA"': 'ZEBRA',

    '!|OFFICIAL_GAME|$_BC|"K_ㄱ_기린"': '기린',
    '!|OFFICIAL_GAME|$_BC|"K_ㄴ_낙타"': '낙타',
    '!|OFFICIAL_GAME|$_BC|"K_ㄷ_다람쥐"': '다람쥐',
    '!|OFFICIAL_GAME|$_BC|"K_ㄹ_라마"': '라마',
    '!|OFFICIAL_GAME|$_BC|"K_ㅁ_물개"': '물개',
    '!|OFFICIAL_GAME|$_BC|"K_ㅂ_반달곰"': '반달곰',
    '!|OFFICIAL_GAME|$_BC|"K_ㅅ_사슴"': '사슴',
    '!|OFFICIAL_GAME|$_BC|"K_ㅇ_앵무새"': '앵무새',
    '!|OFFICIAL_GAME|$_BC|"K_ㅈ_젖소"': '젖소',
    '!|OFFICIAL_GAME|$_BC|"K_ㅊ_치타"': '치타',
    '!|OFFICIAL_GAME|$_BC|"K_ㅋ_코알라"': '코알라',
    '!|OFFICIAL_GAME|$_BC|"K_ㅌ_토끼"': '토끼',
    '!|OFFICIAL_GAME|$_BC|"K_ㅍ_펭귄"': '펭귄',
    '!|OFFICIAL_GAME|$_BC|"K_ㅎ_하마"': '하마',

    '!|OFFICIAL_COMPOSER|$_BC|"DO"':'도',
    '!|OFFICIAL_COMPOSER|$_BC|"RE"': '레',
    '!|OFFICIAL_COMPOSER|$_BC|"MI"': '미',
    '!|OFFICIAL_COMPOSER|$_BC|"FA"': '파',
    '!|OFFICIAL_COMPOSER|$_BC|"SOL"': '솔',
    '!|OFFICIAL_COMPOSER|$_BC|"LA"': '라',
    '!|OFFICIAL_COMPOSER|$_BC|"TI"': '시',
    '!|OFFICIAL_COMPOSER|$_BC|"DO2"': '도2',

    '!|OFFICIAL_WEATHER|_START|""': '날씨',
    '!|OFFICIAL_NEWS|_START|""': '뉴스',
    '!|OFFICIAL_DANCE|_START|""': '댄스',
    '!|OFFICIAL_MUSIC|_START|""': '음악',
    '!|OFFICIAL_EXERCISE|_START|""': '체조',
    '!|OFFICIAL_CAMERA|_START|""': '카메라',

    '!|_PIBO|$_ACTION|"RUN"': '수행',
    '!|_PIBO|$_ACTION|"REPEAT"': '반복',
    '!|_PIBO|$_ACTION|"DIALOG"': '대화',

    '!|_PIBO|$_EYE|"255,0,0"': '빨강',
    '!|_PIBO|$_EYE|"255,128,0"': '주황',
    '!|_PIBO|$_EYE|"255,255,0"': '노랑',
    '!|_PIBO|$_EYE|"0,255,0"': '초록',
    '!|_PIBO|$_EYE|"0,0,255"': '파랑',
    '!|_PIBO|$_EYE|"255,20,147"': '핑크',
    '!|_PIBO|$_EYE|"128,0,128"': '보라',

    '!|_PIBO|$_MOTION|"RIGHT"':'우회전',
    '!|_PIBO|$_MOTION|"LEFT"':'좌회전',
    '!|_PIBO|$_MOTION|"FORWARD"': '전진',
    '!|_PIBO|$_MOTION|"BACKWARD"': '후진',
    '!|_PIBO|$_MOTION|"BORING"': '지루함',
    '!|_PIBO|$_MOTION|"EXCITE"': '신남',
    '!|_PIBO|$_MOTION|"SAD"': '슬픔',
    '!|_PIBO|$_MOTION|"HAPPY"': '기쁨',
  }

  #d = decodeQR('"}`DVSSFOU}%`CD}#1#')
  print(len(test_data))
  for item in test_data:
    res = decode_CARD(item)

    if res == None:
      print("None:", item)
    elif res != test_data[item]:
      print("Err:", item, res, test_data[item])
    else:
      print("Passed: {}", item)