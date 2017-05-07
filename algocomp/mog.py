from template import *

import time,resource
import numpy as np


cap = camera
#help(cv2.BackgroundSubtractorMOG2)
fgbg = cv2.BackgroundSubtractorMOG()

past=time.time()

fps=0

decisions = [0]
frame_counter=0

top_thresh = 20000
bottom_thresh = 20000

while(1):
	present=time.time()
	timeelapsed=present-past
	past=present

	#print 1/timeelapsed

	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	y,u,v = cv2.split(frame)
	frame = y
	frame = cv2.medianBlur(frame,5)

	fgmask = fgbg.apply(frame,learningRate=0.05)
	cv2.imshow('frame',fgmask)
	


	YconstXsum=sum(np.transpose(fgmask))
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
		try:
			int(sys.argv[1])
		except:
			print "sys.argv[1] not camera"
			fps=20
			print "FPS:",fps
			time.sleep(2)
			continue
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


	k = cv2.waitKey(30) & 0xff
	if k == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()