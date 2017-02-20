import cv2


c = cv2.VideoCapture(0)

for x in range(20):c.read()

while True:
  s,f = c.read()
  cv2.imshow("frame",f)
  if cv2.waitKey(1) & 0xff == ord('q'):break

c.release()
cv2.destroyAllWindows()
