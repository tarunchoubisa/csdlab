import cv2
import numpy as np
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

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
xtemp = p0[0]

p0 = p0[:1]

for x in range(3):
  p0=np.append(p0,xtemp)

p0[1],p0[0] = height/4,width/2
p0[3],p0[2] = height/4+height/2/3,width/2
p0[5],p0[4] = height/4+(2*height/2/3),width/2
p0[7],p0[6] = height/4+(3*height/2/3),width/2

p0_backup = np.copy(p0)
mask_blank = np.copy(mask)

def init_points():
  global p0
  p0=p0_backup
  #fill points if less than 4
  p0=p0.reshape(4,1,2)


init_points()
frame_count=0

while True:
  frame_count=frame_count+1

  if frame_count>=20:
    init_points()
    mask = np.copy(mask_blank)
    frame_count=0
    continue


  s1,f1 = cap.read()
  s,frame = cap.read()
  frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  old_gray = frame_gray.copy()

  nDetectedpoints=sum(st)
  print nDetectedpoints

  if nDetectedpoints<4:
    print "--------->","reinit p0"
    mask = np.copy(mask_blank)
    init_points()
    continue

  p1_new = p1[st==1]
  p0_old = p0[st==1]

  #print 'p0' + str(p1)
  #print 'st' + str(st)
  #exit()

  #cv2.imshow("frame",frame)
  for i,(new,old) in enumerate(zip(p1_new,p0_old)): # this syntax allows to take the elements of good_new  in variable new and old
    #print enumerate(zip(good_new,good_old))
    a,b = new.ravel() # good_new, it returns the flattened array, this is the coordinate corresponding to good_new points
    c,d = old.ravel() # good_old, this is the old coordinate
    cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)

  p0 = p1_new.reshape(len(p1_new.reshape(-1))/2,1,2)

  if cv2.waitKey(1) & 0xff == ord('q'):break


cap.release()
cv2.destroyAllWindows()
