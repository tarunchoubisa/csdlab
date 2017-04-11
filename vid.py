#import numpy as np
from time import sleep

import os,subprocess

files = os.listdir("videos")
print "\n".join(files)


for file in files:

	cmd = "xterm -e python " + os.getcwd() +  "/sam.py " + os.getcwd() + "/videos/" + file
	print cmd
	#raw_input()
	try:
		subprocess.check_output(cmd,shell=1)
	except:
		pass
		sleep(3)
	


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