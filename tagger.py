import cv2
import sys,os
import numpy as np
import datetime

tagFolder="TAGS"
if not os.path.exists(tagFolder):
	os.mkdir(tagFolder)


camera_id=sys.argv[1]

try:
  camera_id = int(camera_id)
except:
  pass


cap = cv2.VideoCapture(camera_id)

for i in range(10):
	ret,f=cap.read()
	cv2.imshow('frame',f)
	if cv2.waitKey(1) & 0xff==ord('q'):break


fcount=0
tagcount=0
tagbegin=0

tagval="c"


tagged_data=[]


while 1:
	#print fcount
	fcount+=1
	ret,f=cap.read()
	if not ret:
		break
	cv2.imshow('frame',f)
	if cv2.waitKey(1) & 0xff==ord('q'):break
	while 1:
		print "Enter tag: ",
		ip=raw_input()
		if ip=="d" or ip=='h' or ip=='c' or ip=='ambiguous':
			tagval=ip
			break
		elif ip=="":
			break
		else:
			print "Tag can only be 'd' or 'h' or 'c' or 'ambiguous'"


	print fcount,tagval

	tagged_data.append([fcount,tagval])

with open(tagFolder+"/"+camera_id.split("/")[-1]+"_TAG.txt",'w') as g:
	import json
	g.write(json.dumps(tagged_data))

sys.exit(0)