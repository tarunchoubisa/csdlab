import cv2
import sys

camera = cv2.VideoCapture(int(sys.argv[1]))

while 1:
  ret,frame=camera.read()
  cv2.imshow('f',frame)

  if cv2.waitKey(30) & 0XFF  == ord('q'):
    break


camera.release()
cv2.destroyAllWindows()
