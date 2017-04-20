import cv2 
import numpy as np
import sys
import time
import resource

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

past = time.time()

while 1:
	present = time.time()
	timeelapsed=present-past
	past = present

	#print 1/timeelapsed

	status,frame=camera.read()

	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	 
	diff=cv2.absdiff(lastframe,frame)
	
	###----------------------------- UNCOMMENTING THE FOLLOWING MAKES THIS consequtiveSubtraction ---------------------------
	lastframe=frame

	ret,thresh = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)

	cv2.imshow("window name",thresh)

	YconstXsum=sum(np.transpose(thresh))
	sumlen=YconstXsum.shape[0]
	
	top=YconstXsum[:sumlen/2]
	bottom=YconstXsum[sumlen/2:]

	top,bottom=sum(top),sum(bottom)


	print "Decision",
	if(top>5000):print 'Human',
	elif(bottom>5000):print 'animal',
	else: print "None",
	print " ", 1/timeelapsed ,"",
	print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   
	if cv2.waitKey(1) & 0xff == ord('q'):
		break


camera.release()
cv2.destroyAllWindows()
