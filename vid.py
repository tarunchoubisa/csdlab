#import numpy as np
from time import sleep

import os,subprocess,sys

files = os.listdir("videos")
print "\n".join(files)

for file in files:

	cmd = "xterm -e python " + os.getcwd() +  "/" + sys.argv[1] + " " + os.getcwd() + "/videos/" + file
	print cmd
	#raw_input()
	try:
		output=subprocess.check_output(cmd,shell=1)
	except Exception as e:
		print e
		sleep(3)

	#raw_input()
	

exit()



import cv2

ExceptionFlag=False

for file in files:
	print "--->",file
	cap = cv2.VideoCapture('videos/'+file)
	while(cap.isOpened()):
		ret, frame = cap.read()
		
		try:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imshow('frame',frame)
		except Exception as e:
			ExceptionFlag=True
		if ExceptionFlag==True or cv2.waitKey(1) & 0xFF == ord('q'):
			ExceptionFlag=False
			cap.release()
			cv2.destroyAllWindows()
			break
