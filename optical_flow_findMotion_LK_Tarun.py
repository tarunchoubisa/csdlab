import numpy as np
import cv2
import time
import sys
if len(sys.argv)>1:
  camera_id=1
else:
  camera_id=0# 0 for laptop camera, 1 for an external camera 
#camera_id=1
print camera_id
#raw_input()


###GOBALS
'''cap=0
feature_params=0
lk_params=0'''
#global old_frame, old_gray
global all_not_active
all_not_active=0
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        ix,iy = x,y

def update_initial_points(p0df):
  tol=160
  first_x=np.random.randint(160,160+tol,1)
  second_x =first_x+160 #np.random.randint(320-tol,320+tol,1)
  np.put(p0df,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[first_x,100,first_x,200,second_x,100,second_x,200,first_x,300,first_x,400,second_x,300,second_x,400])
  xsize=len(p0df)
  p0dfr=p0df.reshape(xsize/2,1,2) # reshaping the array to multi-dimensional array
  return p0dfr



def initialization(camera_id):
  #cap = cv2.VideoCapture('15-11-2016-09-56-25-aaaa-0-0-0-200-0-0-7.mp4')
  global cap,feature_params,lk_params, color
  cap = cv2.VideoCapture(camera_id) 
  #params for ShiTomasi corner detection
  feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
                        # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (15,15),
                   maxLevel = 3,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # maxlevel decides the pyramid level 
  #Create some random colors
  color = np.random.randint(0,255,(100,3))
  for i in range(50):
    cap.read()
    print 'wait'
  print 'If AttributeError: NoneType object has no attribute shape, then check the camera_id in the begining \n'  
  print 'If camera_id is correct and still giving the problem then connect and disconnect the USB \n'

def read_a_gray_image():
  #Take first frame and find corners in it
  
  ret, old_frame = cap.read()
  #print type(old_frame)
  #height, width, channel = old_frame.shape
  #print height, width
  old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  return old_frame, old_gray
#old_gray.shape()
#print type(old_gray)

def GetPointsToTrack():
  global points_to_keep, p0dfrs,p0df,p0dfr
  p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  print 'p0' + str(p0)
  p0s =  p0.shape
  points_to_keep=8
  p0std=p0s[0]-points_to_keep # total_size- points_to_keep
  #p0d=np.delete(p0,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46),axis=0)
  #p0d=np.delete(p0,(0,1,2,3,4),axis=0)
  p0d=np.delete(p0,np.arange(p0std),axis=0) # will remove the data from array except no. of points_to_keep
  np.arange(3)
  p0ds=p0d.shape
  print' p0ds_shape: ' + str(p0ds)
  print 'p0ds_type: ' + str(type(p0d))
  #exit()
  p0df= p0d.flatten() # it makes from array of tupples to 1D array
  #print p0df
  #print 'p0df: ' + str(p0df.shape )
  h=173
  #Puting our own features in the form of 1D array, first vector is the index, second vector has the points in the form of pairs 
   #later this 1D array will be reshaped to a 2D array (if 11 points are left, then it will have 22 numbers)
   #np.put(p0df, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [9,375,18,375,36,375,72,375,144,375,288,375,480,375,9,h,27,h,75,h,210,h])
   #########np.put(p0df,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[160,100,160,200,320,100,320,200,160,300,160,400,320,300,320,400])
   #np.put(p0df,[0,1,2,3],[160,80,160,320]) # first half is upper (x,y), y starts from the top
  #print p0df.shape
  p0df = p0df[:1]
  xtemp = p0df[0]
  for x in range(16):
    p0df=np.append(p0df,xtemp)
  p0df= p0df[1:]
  xsize=len(p0df)
  first_x=160
  second_x =320
  np.put(p0df,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[first_x,100,first_x,200,second_x,100,second_x,200,first_x,300,first_x,400,second_x,300,second_x,400])
  #p0dfr=p0df.reshape(p0ds[0],p0ds[1],p0ds[2]) # reshaping the array to multi-dimensional array 
  p0dfr=p0df.reshape(xsize/2,1,2) # reshaping the array to multi-dimensional array 
  #print 'p0dfr'+ str(p0dfr)
  #print 'p0dfr_shape' + str(p0dfr.shape)
  #exit() 
  #print p0.ndim
  #p0dfrs=np.vstack((p0,p0dfr))
  print 'p0dfr_type' + str(type(p0dfr))
  print 'p0dfr: ' + str(p0dfr)
  p0dfrs=p0dfr # old points, helpful when new point location will  be estimated by optical flow and line has to be drawn between old and new

def TraceInitialization():
  global mask, points_init, points_final, time_to_calc_distance, DiffUpperSum, DiffLowerSum
  mask = np.zeros_like(old_frame)
  #print type(mask) 
  #cv2.imshow('frame1',mask)
  #p0b=np.vstack((p0,b))
  points_init=np.array([(0,0)]*points_to_keep)
  points_final = np.array([(0,0)]*points_to_keep)
  time_to_calc_distance = 20
  DiffUpperSum=0
  DiffLowerSum=0

def reinitialize_if_less_active(st,p0df):
  #print 'sum_st' + str(sum(st))
  if sum(st)<8:
    update_initial_points(p0df) # update the initial points 
    p0dfrs = p0dfr
    #print 'sum_st' + str(sum(st))
    #print 'p0dfrs-shape' + str(p0dfrs)
    all_not_active==1
    return all_not_active
    print 'all not active in loop ' + str(all_not_active)

def get_good_new_old_points(p1,p0dfrs,st):
  global good_new, good_old, good_new_shape
# Select new good point coordinates and their corresponding old locations to draw the line
  #print 'p1-type: ' + str(type(p1))
  p1s=p1.shape
  #print 'p1-shape ' + str(p1s)
  good_new = p1[st==1] 
  good_old = p0dfrs[st==1] # this is not in a while loop, it is the same as first time it has taken 
  good_new_shape=good_new.shape
  #print 'good_new_shape' + str(good_new_shape[0])
 # print 'p0dfrs_first' + str(p0dfrs)

def get_initial_final_points_mask(good_new,good_old):
  for i,(new,old) in enumerate(zip(good_new,good_old)): # this syntax allows to take the elements of good_new  in variable new and old
    #print enumerate(zip(good_new,good_old))
    a,b = new.ravel() # good_new, it returns the flattened array, this is the coordinate corresponding to good_new points
    c,d = old.ravel() # good_old, this is the old coordinate
    cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    if count_flag==1:
      points_init[i] = (a,b) # new points
    elif count_flag==20:
      points_final[i] = (c,d) # old points

def draw_track(frame,mask):
   ########################################################### draw the tracks
  #print 'points_to_keep by 2: ' + str(points_to_keep/2)
  #print 'points_final_j' + str(points_final) + '\n'
  img = cv2.add(frame,mask)
  cv2.imshow('frame',img)
 # cv2.setMouseCallback('frame',draw_circle)
def update_old_frame_marker_points(frame_gray, good_new):
  # Now update the previous frame and previous points
  global old_gray,p0dfrs
  old_gray = frame_gray.copy()
  p0dfrs = good_new.reshape(-1,1,2) # p0dfrs is changed here 

  #print 'good_old' + str(good_old)
  #print 'good_new' + str(good_new) 
  #print 'p0dfrs' + str(p0dfrs)+'\n'
    ################################### main

initialization(camera_id)
old_frame, old_gray = read_a_gray_image() 
GetPointsToTrack() # Feature to Track  (But we delet all these and put our own features)
TraceInitialization()# Initializing the parameters to trace the points 

#  Forever while loop
#print 'entring while loop'
count_flag=0;
while(1):
  current_milli_time = lambda: int(round(time.time()*1000))
  tic=current_milli_time()
  #print 'mask_shape' + str(mask.shape)
  #print 'p0dfr: ' + str(p0dfr)
  frame,frame_gray = read_a_gray_image() # get the frame for current sample/time
# calculate optical flow

  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0dfrs, None, **lk_params) # st==1 denotes the good points(traceable)
  # p1 are new estimated points
  #print 'p0df-shape' + str(p0df.shape)
  all_not_active = reinitialize_if_less_active(st,p0df)
  if all_not_active:
    continue # if it doesn't give all the coordinates then it will not find any distance and will not give any decision too'''
  print 'sum_st' + str(sum(st))
  print 'p0dfrs' + str(p0dfrs)
  #raw_input()
  get_good_new_old_points(p1,p0dfrs,st) # active points, now all will be active points, otherwise it will not reach here
  
  count_flag=count_flag+1 
  get_initial_final_points_mask(good_new,good_old)# find out the initial and final points at some time interval
  draw_track(frame,mask)# adds the frame and mask and show
  update_old_frame_marker_points(frame_gray, good_new) 

  toc=current_milli_time()
  exact_time=toc-tic
  diff = points_final - points_init
  #print 'count_flag' + str(count_flag)
  #print 'points_final_old_points' + str(points_final)
  #print 'points_init_new_points' + str(points_init)
  diff = diff**2
  #print 'exact_time: ' + str(exact_time)
  if count_flag==20:
    for j  in range(good_new_shape[0]):
      if points_final[j][1]<240: # Upper half
        DiffUpperSum = DiffUpperSum + sum(diff[j][:])
      else:
        DiffLowerSum = DiffLowerSum + sum(diff[j][:])
    count_flag=0
    mask.fill(0)
    print 'DiffUpperSum		' + str(DiffUpperSum) + '	DiffLowerSum	' + str(DiffLowerSum) + '\n'
    #print 'points_final' + str(points_final)
    points_init.fill(0)
    points_final.fill(0)
    if DiffUpperSum>1000 and DiffLowerSum>1000:
      print "--------------------------------------------"*2,'Human Detected'
    elif DiffLowerSum>1000:
      print '--------------------------------------------'*2,'Animal Detected'
    p0dfrs=p0dfr
    DiffUpperSum=0
    DiffLowerSum=0
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
cv2.destroyAllWindows()
cap.release()
