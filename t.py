import cv2
import sys
c = cv2.VideoCapture(int(sys.argv[1]))

y=0

while True:
  print "init",y
  x,f=c.read()
  print "end",y
  y= y+1
  cv2.imshow('frame',f)
  if cv2.waitKey(1)&0xff==ord('q'):break


c.release()
cv2.destroyAllWindows()
