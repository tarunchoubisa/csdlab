import numpy as np
import cv2
import time

camera_id=0;# 0 for laptop camera, 1 for an external camera 
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        ix,iy = x,y

#cap = cv2.VideoCapture('15-11-2016-09-56-25-aaaa-0-0-0-200-0-0-7.mp4')
cap = cv2.VideoCapture(camera_id) 
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
 
 # Take first frame and find corners in it

ret, old_frame = cap.read()
#print type(old_frame)
height, width, channel = old_frame.shape
#print height, width
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#old_gray.shape()
#print type(old_gray)

 ############################################################ Feature to Track  (But we delet all these and put our own features)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print 'p0' + str(p0)
p0s =  p0.shape
p0std=p0s[0]-8 # total_size- points_to_keep
#p0d=np.delete(p0,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46),axis=0)
#p0d=np.delete(p0,(0,1,2,3,4),axis=0)
p0d=np.delete(p0,np.arange(p0std),axis=0) # will remove the data from array except no. of points_to_keep
np.arange(3)
p0ds=p0d.shape
#print' p0ds' + str(p0ds)
p0df= p0d.flatten() # it makes from array of tupples to 1D array
#print p0df
print p0df.shape 
h=173

 # Puting our own features in the form of 1D array, first vector is the index, second vector has the points in the form of pairs 
# later this 1D array will be reshaped to a 2D array (if 11 points are left, then it will have 22 numbers)
#np.put(p0df, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [9,375,18,375,36,375,72,375,144,375,288,375,480,375,9,h,27,h,75,h,210,h])
np.put(p0df,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[160,100,320,100,160,200,320,200,160,300,320,300,160,400,320,400])
#np.put(p0df,[0,1],[375,375])
#print p0df.shape
p0dfr=p0df.reshape(p0ds[0],p0ds[1],p0ds[2]) # reshaping the array to multi-dimensional array 
#print 'p0dfr'+ str(p0dfr)
#print 'p0dfr_shape' + str(p0dfr.shape)
#exit() 
#print p0.ndim
#p0dfrs=np.vstack((p0,p0dfr))
p0dfrs=p0dfr # old points, helpful when new point location will  be estimated by optical flow and line has to be drawn between old and new

############################### Create a mask image for drawing purposes

mask = np.zeros_like(old_frame)
#print type(mask) 
#cv2.imshow('frame1',mask)
#p0b=np.vstack((p0,b))
current_milli_time = lambda: int(round(time.time()*1000))
count_flag=0;
################################ Forever while loop
while(1):
  tic=current_milli_time()
  ret,frame = cap.read()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
# calculate optical flow

  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0dfrs, None, **lk_params) # st==1 denotes the good points(traceable)
  # p1 are new estimated points
    
# Select new good point coordinates and their corresponding old locations to draw the line

  good_new = p1[st==1] 
  good_old = p0dfrs[st==1]

########################################################### draw the tracks
  count_flag=count_flag+1 
  a_init=[]
  b_init=[]
  a_final=[]
  b_final=[]     
  for i,(new,old) in enumerate(zip(good_new,good_old)): # this syntax allows to take the elements of good_new  in variable new and old
    #print enumerate(zip(good_new,good_old))
    a,b = new.ravel() # it returns the flattened array, this is the coordinate corresponding to good_new points
    c,d = old.ravel() # this is the old coordinate
    cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    if count_flag==1:
      a_init[i]=a
      b_init[i]=b
    elif count_flag==100:
      a_final[i]=a
      b_final[i]=b
       
  img = cv2.add(frame,mask)
  cv2.imshow('frame',img)
 # cv2.setMouseCallback('frame',draw_circle)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
 
# Now update the previous frame and previous points

  old_gray = frame_gray.copy()
  p0dfrs = good_new.reshape(-1,1,2)
  toc=current_milli_time()
  exact_time=toc-tic
  print 'exact_time: ' + str(exact_time)
cv2.destroyAllWindows()
cap.release()
