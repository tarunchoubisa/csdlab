import cv2
import numpy as np


c = cv2.VideoCapture(0)

for x in range(20):c.read()

s,frame = c.read()
old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

height,width,channels = frame.shape


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

'''
ys = range(height/4,height - height/4+1,height/2/3)
#ys = range(120,361,80)
xs = [width/2]*4
print height,width

#mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
p0 = zip(ys,xs)
p0 = np.array(map(lambda p:[map(lambda a:float(a),p)],p0))
print p0.shape

#print p0
'''

'''for p in p0:
	mask[p] = 250

while True:
	cv2.imshow('mask',mask)
	if cv2.waitKey(1) & 0xff == ord('q'):break
'''

# Take first frame and find corners in it
ret, old_frame = c.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = p0[:8]
#print p0.shape

#exit()

while True:
  s,f = c.read()
  frame_gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  old_gray = frame_gray.copy()
  
  p1 = p1[st==1]
  p0 = p0[st==1]

  #print p0,p1

  print st

  cv2.imshow("frame",f)

  if cv2.waitKey(1) & 0xff == ord('q'):break

c.release()
cv2.destroyAllWindows()
