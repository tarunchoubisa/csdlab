from template import *

import time


cap = camera
#help(cv2.BackgroundSubtractorMOG2)
fgbg = cv2.BackgroundSubtractorMOG()

past=time.time()

while(1):
	present=time.time()
	elapsed=present-past
	past=present

	print 1/elapsed

	ret, frame = cap.read()
	fgmask = fgbg.apply(frame,learningRate=0.05)
	cv2.imshow('frame',fgmask)
	k = cv2.waitKey(30) & 0xff
	if k == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()