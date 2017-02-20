import cv2
import numpy as np
# hi how areyou
#iam just okay
# another hi comment
camera_id=0
cap = cv2.VideoCapture(camera_id)
for x in range(20):
  cap.read()
s,frame = cap.read()
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
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = p0[:8]
print 'p0_shape' + str(p0.shape)
print 'p0' + str(p0)
#exit()

while True:
  s1,f1 = cap.read()
  s,frame = cap.read()
  frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  old_gray = frame_gray.copy()
  
  p1_new = p1[st==1]
  p0_old = p0[st==1]

  print 'p0' + str(p0)
  print 'st' + str(st)

  #cv2.imshow("frame",frame)
  for i,(new,old) in enumerate(zip(p1_new,p0_old)): # this syntax allows to take the elements of good_new  in variable new and old
    #print enumerate(zip(good_new,good_old))
    a,b = new.ravel() # good_new, it returns the flattened array, this is the coordinate corresponding to good_new points
    c,d = old.ravel() # good_old, this is the old coordinate
    cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)

  if cv2.waitKey(1) & 0xff == ord('q'):break

c.release()
cv2.destroyAllWindows()
