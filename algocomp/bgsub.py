import cv2 

import sys

cam_source = sys.argv[1]


try:
	cam_source=int(cam_source)
except:
	pass

camera = cv2.VideoCapture(cam_source)

for x in range(20):
	camera.read()


status,lastframe=camera.read()
lastframe = cv2.cvtColor(lastframe,cv2.COLOR_BGR2GRAY)

while 1:
	status,frame=camera.read()

	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	 
	diff=cv2.absdiff(lastframe,frame)
	
	cv2.imshow("window name",diff)


	if cv2.waitKey(1) & 0xff == ord('q'):
		break


camera.release()
cv2.destroyAllWindows()
