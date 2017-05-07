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
#lastframe = cv2.cvtColor(lastframe,cv2.COLOR_BGR2GRAY)
lastframe = cv2.cvtColor(lastframe, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(lastframe)
lastframe = y
lastframe = cv2.medianBlur(lastframe,5)

past = time.time()

fps=0

decisions = [0]
frame_counter=0

top_thresh = 10000
bottom_thresh = 10000

while 1:
	present = time.time()
	timeelapsed=present-past
	past = present

	#print 1/timeelapsed

	status,frame=camera.read()

	#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	y,u,v = cv2.split(frame)
	frame = y
	frame = cv2.medianBlur(frame,5)
	 
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
	if(top>top_thresh):
		current_decision=1
		print 'Human',
	elif(bottom>bottom_thresh):
		current_decision=-1
		print 'animal',
	else:
		current_decision=0
		print "None",
	print " ", 1/timeelapsed ,"",
	print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000,'MBytes'

	if not fps:
		fps = int(1/timeelapsed)
		if fps>0 and fps<30:
			pass
		else:
			fps=0
			continue
		decisions=decisions*fps
		print "FPS:",fps
		time.sleep(2)

	decisions.pop(0)
	decisions.append(current_decision)

	frame_counter+=1
	if frame_counter>=fps:
		frame_counter=0
		print decisions
		voted_decision=sum(decisions)
		print "----------------------",
		if voted_decision==0:
			print "None"
		elif voted_decision>0:
			print "Human"
		elif voted_decision<0:
			print "Animal"


	if cv2.waitKey(30) & 0xff == ord('q'):
		break


camera.release()
cv2.destroyAllWindows()
