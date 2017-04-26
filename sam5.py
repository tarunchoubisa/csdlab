import cv2
import sys
import numpy as np
import datetime,time

from matplotlib import pyplot as plt

camera_id=sys.argv[1]

try:
  camera_id = int(camera_id)
except:
  pass


cap = cv2.VideoCapture(camera_id)

for x in range(50):
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

for x in range(7):
  p0=np.append(p0,xtemp)

p0[1],p0[0] = height/4,width/3
p0[3],p0[2] = height/4+height/2/3,width/3
p0[5],p0[4] = height/4+(2*height/2/3),width/3
p0[7],p0[6] = height/4+(3*height/2/3),width/3

p0[8+1],p0[8+0] = height/4,2*width/3
p0[8+3],p0[8+2] = height/4+height/2/3,2*width/3
p0[8+5],p0[8+4] = height/4+(2*height/2/3),2*width/3
p0[8+7],p0[8+6] = height/4+(3*height/2/3),2*width/3


p0_backup = np.copy(p0)
mask_blank = np.copy(mask)
p0_backup_preshaped = p0_backup.reshape(8,1,2)


def init_points():
  global p0
  p0=p0_backup
  #fill points if less than 4
  p0=p0.reshape(8,1,2)



#########################

FeedSizeMAX=100

DistFeedsLA=[]
DistFeedsLB=[]
DistFeedsLC=[]
DistFeedsLD=[]
DistFeedsRA=[]
DistFeedsRB=[]
DistFeedsRC=[]
DistFeedsRD=[]

DispFeedsLA=[]
DispFeedsLB=[]
DispFeedsLC=[]
DispFeedsLD=[]
DispFeedsRA=[]
DispFeedsRB=[]
DispFeedsRC=[]
DispFeedsRD=[]

def feed(disp):
  feed.size+=1

  #print feed.size
  #raw_input()
  
  dispLA = disp[0][0]
  dispLB = disp[1][0]
  dispLC = disp[2][0]
  dispLD = disp[3][0]

  dispRA = disp[4][0]
  dispRB = disp[5][0]
  dispRC = disp[6][0]
  dispRD = disp[7][0]

  #print dispLA,dispLB,dispLC,dispLD
  #print dispRA,dispRB,dispRC,dispRD
  
  DispFeedsLA.append(dispLA)
  DispFeedsLB.append(dispLB)
  DispFeedsLC.append(dispLC)
  DispFeedsLD.append(dispLD)

  DispFeedsRA.append(dispRA)
  DispFeedsRB.append(dispRB)
  DispFeedsRC.append(dispRC)
  DispFeedsRD.append(dispRD)

  dist = disp.copy()
  dist = np.sqrt(dist*dist)

  distLA = sum(dist[0][0])
  distLB = sum(dist[1][0])
  distLC = sum(dist[2][0])
  distLD = sum(dist[3][0])

  distRA = sum(dist[4][0])
  distRB = sum(dist[5][0])
  distRC = sum(dist[6][0])
  distRD = sum(dist[7][0])

  DistFeedsLA.append(distLA)
  DistFeedsLB.append(distLB)
  DistFeedsLC.append(distLC)
  DistFeedsLD.append(distLD)

  DistFeedsRA.append(distRA)
  DistFeedsRB.append(distRB)
  DistFeedsRC.append(distRC)
  DistFeedsRD.append(distRD)

  if feed.size>FeedSizeMAX:
    feed.size-=1
    DispFeedsLA.pop(0)
    DispFeedsLB.pop(0)
    DispFeedsLC.pop(0)
    DispFeedsLD.pop(0)
    DispFeedsRA.pop(0)
    DispFeedsRB.pop(0)
    DispFeedsRC.pop(0)
    DispFeedsRD.pop(0)

    DistFeedsLA.pop(0)
    DistFeedsLB.pop(0)
    DistFeedsLC.pop(0)
    DistFeedsLD.pop(0)
    DistFeedsRA.pop(0)
    DistFeedsRB.pop(0)
    DistFeedsRC.pop(0)
    DistFeedsRD.pop(0)

 #print (feed.size),len(DispFeedsLA)
feed.size=0



def VectorCorr(A,B,normalise=1):
  vectLen=len(A[0]) #take first element of A and check its length
  A=np.array(A)
  B=np.array(B)

  #length of correlation array = 2*lenOfDataArray - 1
  vectorInnerProductCorrelation=np.array([0.0]*(2*len(A)-1))

  #correlation of vectors by inner(dot) product is same
  # as sum of correlation of corresponding components of the vetor
  for column in range(vectLen):
    #print A[:,column]
    #print B[:,column]
    #raw_input()
    vectorInnerProductCorrelation += np.correlate(A[:,column],B[:,column],'full')

  if normalise==0:
    return vectorInnerProductCorrelation

  energyA=sum(sum(A*A))
  energyB=sum(sum(B*B))

  normalisedCorr = vectorInnerProductCorrelation / np.sqrt(energyA * energyB)

  return normalisedCorr


def Corr(A,B,normalise=1):
  A=np.array(A)
  B=np.array(B)

  corr=np.correlate(A,B,'full')

  if normalise==0:
    return corr

  energyA=sum(A*A)
  energyB=sum(B*B)

  corr=corr/np.sqrt(energyA*energyB)

  return corr
#########################


init_points()
frame_count=0
decisionFrameCount=0

fps=0
past=time.time()
time.sleep(0.1)


while True:
  #time.sleep(0.02)
  frame_count=frame_count+1
  decisionFrameCount=decisionFrameCount+1
  

  present = time.time()
  fps = present-past
  fps = 1/fps
  #print "FPS:",fps
  past = present


  #time.sleep(0.2)
  s,frame = cap.read()
  frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  old_gray = frame_gray.copy()

  nDetectedpoints=sum(st)
  #print nDetectedpoints==4

  if nDetectedpoints<8:
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

  disp=(p1-p0_backup_preshaped)
  #dist = dist*dist

  p0 = p1_new.reshape(len(p1_new.reshape(-1))/2,1,2)

  if cv2.waitKey(1) & 0xff == ord('q'):
  	break


  feed(disp)

  #print frame_count,decisionFrameCount


  if frame_count>=15:
    init_points()
    mask = np.copy(mask_blank)
    frame_count=0

    if decisionFrameCount<=30:
    	continue
    else:
    	decisionFrameCount=0
    	pass


    #DispFeedsLAB=
    #DispFeedsRAB=

    #DispFeedsLCD=
    #DispFeedsRCD=

    VcorrA=VectorCorr(DispFeedsLA,DispFeedsRA,normalise=1)
    EcorrA=Corr(DistFeedsLA,DistFeedsRA)

    VcorrB=VectorCorr(DispFeedsLB,DispFeedsRB,normalise=1)
    EcorrB=Corr(DistFeedsLB,DistFeedsRB)

    VcorrC=VectorCorr(DispFeedsLC,DispFeedsRC,normalise=1)
    EcorrC=Corr(DistFeedsLC,DistFeedsRC)

    VcorrD=VectorCorr(DispFeedsLD,DispFeedsRD,normalise=1)
    EcorrD=Corr(DistFeedsLD,DistFeedsRD)


    plt.subplot(4,2,1)
    plt.axis([0,200,-1,1])
    va,=plt.plot(range(len(VcorrA)),VcorrA,'r')
    plt.legend([va],["VA"])
    #plt.show()

    plt.subplot(4,2,3)
    plt.axis([0,200,-1,1])
    vb,=plt.plot(range(len(VcorrB)),VcorrB,'b')
    plt.legend([vb],["VB"])

    #plt.show()

    plt.subplot(4,2,5)
    plt.axis([0,200,-1,1])
    vc,=plt.plot(range(len(VcorrC)),VcorrC,'g')
    plt.legend([vc],["VC"])
    #plt.show()

    plt.subplot(4,2,7)
    plt.axis([0,200,-1,1])
    vd,=plt.plot(range(len(VcorrD)),VcorrD,'c')
    plt.legend([vd],["VD"])


    
    plt.subplot(4,2,2)
    plt.axis([0,200,-1,1])
    ea,=plt.plot(range(len(EcorrA)),EcorrA,'r')
    plt.legend([ea],["EA"])


    plt.subplot(4,2,4)
    plt.axis([0,200,-1,1])
    eb,=plt.plot(range(len(EcorrB)),EcorrB,'b')
    plt.legend([eb],["EB"])


    plt.subplot(4,2,6)
    plt.axis([0,200,-1,1])
    ec,=plt.plot(range(len(EcorrC)),EcorrC,'g')
    plt.legend([ec],["EC"])


    plt.subplot(4,2,8)
    plt.axis([0,200,-1,1])
    ed,=plt.plot(range(len(EcorrD)),EcorrD,'c')
    plt.legend([ed],["ED"])



    plt.show()


    continue

    


print "END"
#raw_input()

cap.release()
cv2.destroyAllWindows()
