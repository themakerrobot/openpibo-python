from openpibo.vision import Camera, Detect

camera = Camera()
detect = Detect()

while True:
  res = detect.detect_pose(camera.read())
  print(detect.analyze_pose(res))

  camera.imwrite("/home/pi/test.jpg", res['img'])
  camera.imshow_to_ide("/home/pi/test.jpg")
