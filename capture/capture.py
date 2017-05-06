import numpy as np
import cv2,sys,os

try:
	camSRC=int(sys.argv[1])
except:
	camSRC=sys.argv[1]

opfile= 'store/' + sys.argv[2] + '.avi'

if os.path.exists(opfile):
	print "File already exists.... Overwrite? Y/n"
	ans=raw_input().strip()
	if ans=='Y':
		pass
	else:
		sys.exit(0)


cap = cv2.VideoCapture(camSRC)

for i in range(20):
	ret,frame=cap.read()

mask = np.zeros_like(frame)

height,width,channels = frame.shape
p0 = [0]*16

p0[1],p0[0] = height/4,width/3
p0[3],p0[2] = height/4+height/2/3,width/3
p0[5],p0[4] = height/4+(2*height/2/3),width/3
p0[7],p0[6] = height/4+(3*height/2/3),width/3

p0[8+1],p0[8+0] = height/4,2*width/3
p0[8+3],p0[8+2] = height/4+height/2/3,2*width/3
p0[8+5],p0[8+4] = height/4+(2*height/2/3),2*width/3
p0[8+7],p0[8+6] = height/4+(3*height/2/3),2*width/3

color = np.random.randint(0,255,(100,3))

#cv2.circle(mask,(447,63), 63, (0,0,255), -1)


for i in range(8):
	cv2.circle(mask,(p0[2*i+0],p0[2*i+1]),7,(0,0,255),-1)

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter(opfile,fourcc, 25.0, (640,480))


try:
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if ret==True:
	        #frame = cv2.flip(frame,0)

	        # write the flipped frame
	        out.write(frame)
	        frame=cv2.add(frame,mask)
	        cv2.imshow('frame',frame)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break
	    else:
	        break

except KeyboardInterrupt:
	print "Gracefully shuting down..."
except Exception as e:
	print e

finally:	
	# Release everything if job is finished
	cap.release()
	out.release()
	cv2.destroyAllWindows()
