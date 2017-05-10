import cv2
import sys,os,json
import numpy as np
import datetime,time,json

from matplotlib import pyplot as plt

camera_id=sys.argv[1]

try:
  camera_id = int(camera_id)
except:
  pass


cap = cv2.VideoCapture(camera_id)

for x in range(9):
  cap.read()


s,frame = cap.read()

### FOR CONSECUTIVE SUBTRACTION ###

lastframe=frame
#lastframe = cv2.cvtColor(lastframe,cv2.COLOR_BGR2GRAY)
lastframe = cv2.cvtColor(lastframe, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(lastframe)
lastframe = y
lastframe = cv2.medianBlur(lastframe,5)

ht,wt=y.shape


frames_elapsed=0
feat_vec=[]


#tagFolder="bgsubTAGS"


#if not os.path.exists(tagFolder):
#  os.mkdir(tagFolder)

### FOR CONSECUTIVE SUBTRACTION ###


old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

height,width,channels = frame.shape


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (31,31),
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

FeedSizeMAX=150

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


BGSubFeeds=[]

def feed(disp,bgsubfeature):
  feed.size+=1

  #print feed.size
  #raw_input()

  BGSubFeeds.append(bgsubfeature)
  
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

    BGSubFeeds.pop(0)

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


frames2skip=0


FRAMES_ELAPSED=0  # TAKE CARE OF THE FRAMES WASTED FROM THE VIDEO IN THE BEGINGING- currently 10

while True:
  time.sleep(0.01)
  FRAMES_ELAPSED+=1
  frame_count=frame_count+1
  decisionFrameCount=decisionFrameCount+1
  

  present = time.time()
  fps = present-past
  fps = 1/fps
  #print "FPS:",fps
  past = present


  #time.sleep(0.2)
  s,frame = cap.read()
  if not s:
    break

  ### BG SUBTRATION ###
  yuvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
  y,u,v = cv2.split(yuvframe)
  yframe = cv2.medianBlur(y,5)
  ydiff=cv2.absdiff(lastframe,yframe)

  ###----------------------------- UNCOMMENTING THE FOLLOWING MAKES THIS consequtiveSubtraction ---------------------------
  lastframe=yframe

  ret,ythresh = cv2.threshold(ydiff,50,255,cv2.THRESH_BINARY)

  cv2.imshow("yframe from yuv",yframe)
  cv2.imshow("thresh",ythresh)



  #########################################################
  #                           #                           #
  #             LA            #           RA              #
  #                           #                           #
  #########################################################
  #                           #                           #
  #             LB            #           RB              #
  #                           #                           #
  #########################################################
  #                           #                           #
  #             LC            #           RC              #
  #                           #                           #
  #########################################################
  #                           #                           #
  #             LD            #           RD              #
  #                           #                           #
  #########################################################


  frame2slice=ythresh

  sliceLA=frame2slice[:ht/4,:wt/2]
  sliceRA=frame2slice[:ht/4,wt/2:]

  sliceLB=frame2slice[ht/4:ht/2,:wt/2]
  sliceRB=frame2slice[ht/4:ht/2,wt/2:]

  sliceLC=frame2slice[ht/2:3*ht/4,:wt/2]
  sliceRC=frame2slice[ht/2:3*ht/4,wt/2:]

  sliceLD=frame2slice[3*ht/4:,:wt/2]
  sliceRD=frame2slice[3*ht/4:,wt/2:]

  cv2.imshow("sliceLA",sliceLA)
  cv2.imshow("sliceRA",sliceRA)

  cv2.imshow("sliceLB",sliceLB)
  cv2.imshow("sliceRB",sliceRB)

  cv2.imshow("sliceLC",sliceLC)
  cv2.imshow("sliceRC",sliceRC)

  cv2.imshow("sliceLD",sliceLD)
  cv2.imshow("sliceRD",sliceRD)


  sliceELA=sum(sum(sliceLA))
  sliceERA=sum(sum(sliceRA))

  sliceELB=sum(sum(sliceLB))
  sliceERB=sum(sum(sliceRB))

  sliceELC=sum(sum(sliceLC))
  sliceERC=sum(sum(sliceRC))

  sliceELD=sum(sum(sliceLD))
  sliceERD=sum(sum(sliceRD))


  BGSUBfeatr={'sliceELA':sliceELA,'sliceERA':sliceERA,'sliceELB':sliceELB,'sliceERB':sliceERB,'sliceELC':sliceELC,'sliceERC':sliceERC,'sliceELD':sliceELD,'sliceERD':sliceERD}

  #for key in featr:
  #  print key,featr[key]



  ### BG SUBTRATION ###



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


  feed(disp,BGSUBfeatr)

  #print frame_count,decisionFrameCount

  #time.sleep(0.1)

  maxDispXaxis=max(np.abs(disp[:,0,0]))
  if maxDispXaxis>100 or frame_count>=FeedSizeMAX:
    #raw_input()
    init_points()
    mask = np.copy(mask_blank)
    frame_count=0

  #make this if redundant
  if True or frame_count>=15:
    #init_points()
    #mask = np.copy(mask_blank)
    #frame_count=0

    if decisionFrameCount<30:
    	continue
    else:
    	decisionFrameCount=0
    	pass


    DispFeedsLAB=np.append(np.array(DispFeedsLA),np.array(DispFeedsLB),axis=1)
    DispFeedsRAB=np.append(np.array(DispFeedsRA),np.array(DispFeedsRB),axis=1)

    DispFeedsLCD=np.append(np.array(DispFeedsLC),np.array(DispFeedsLD),axis=1)
    DispFeedsRCD=np.append(np.array(DispFeedsRC),np.array(DispFeedsRD),axis=1)

    VcorrAB=VectorCorr(DispFeedsLAB,DispFeedsRAB)
    unVcorrAB=VectorCorr(DispFeedsLAB,DispFeedsRAB,normalise=0)
    VcorrCD=VectorCorr(DispFeedsLCD,DispFeedsRCD)
    unVcorrCD=VectorCorr(DispFeedsLCD,DispFeedsRCD,normalise=0)
    

    VcorrA=VectorCorr(DispFeedsLA,DispFeedsRA,normalise=1)
    unVcorrA=VectorCorr(DispFeedsLA,DispFeedsRA,normalise=0)
    EcorrA=Corr(DistFeedsLA,DistFeedsRA)
    unEcorrA=Corr(DistFeedsLA,DistFeedsRA,normalise=0)

    VcorrB=VectorCorr(DispFeedsLB,DispFeedsRB,normalise=1)
    unVcorrB=VectorCorr(DispFeedsLB,DispFeedsRB,normalise=0)
    EcorrB=Corr(DistFeedsLB,DistFeedsRB)
    unEcorrB=Corr(DistFeedsLB,DistFeedsRB,normalise=0)

    VcorrC=VectorCorr(DispFeedsLC,DispFeedsRC,normalise=1)
    unVcorrC=VectorCorr(DispFeedsLC,DispFeedsRC,normalise=0)
    EcorrC=Corr(DistFeedsLC,DistFeedsRC)
    unEcorrC=Corr(DistFeedsLC,DistFeedsRC,normalise=0)

    VcorrD=VectorCorr(DispFeedsLD,DispFeedsRD,normalise=1)
    unVcorrD=VectorCorr(DispFeedsLD,DispFeedsRD,normalise=0)
    EcorrD=Corr(DistFeedsLD,DistFeedsRD)
    unEcorrD=Corr(DistFeedsLD,DistFeedsRD,normalise=0)


    '''

    if not frames2skip==0:
      print "skipping frames",frames2skip
      frames2skip -= 1
      continue
    else:
      pass

    

    plt.figure(figsize=(10,10))

    
    plt.subplot(4,2,1)
    plt.axis([0,200,-1,1])
    #plt.xlim(0,len(VcorrA))
    va,=plt.plot(range(len(VcorrA)),VcorrA,'r')
    plt.legend([va],["VA"])
    #plt.show()

    plt.subplot(4,2,3)
    plt.axis([0,200,-1,1])
    #plt.xlim(0,len(VcorrB))

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
    #plt.xlim(0,len(EcorrA))

    ea,=plt.plot(range(len(EcorrA)),EcorrA,'r')
    plt.legend([ea],["EA"])


    plt.subplot(4,2,4)
    plt.axis([0,200,-1,1])
    #plt.xlim(0,len(EcorrB))

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
    
    try:
      plt.show()
    except Exception as e:
      print e

    

    plt.figure(figsize=(10,10))

    plt.subplot(10,1,1)
    #plt.axis([0,200,-1,1])
    plt.ylim(-1,1)
    vab,=plt.plot(range(len(VcorrAB)),VcorrAB,'r')
    plt.legend([vab],["VAB"])

    plt.subplot(10,1,2)
    #plt.axis([0,200,-1,1])
    plt.ylim(-1,1)
    vcd,=plt.plot(range(len(VcorrCD)),VcorrCD,'b')
    plt.legend([vcd],["VCD"])

    plt.subplot(10,1,3)
    #plt.axis([0,200,-1,1])
    alx,=plt.plot(range(len(DispFeedsLA)),np.array(DispFeedsLA)[:,0],'k')
    arx,=plt.plot(range(len(DispFeedsRA)),np.array(DispFeedsRA)[:,0],'c')
    plt.legend([alx,arx],["alx","arx"])
    
    plt.subplot(10,1,4)
    aly,=plt.plot(range(len(DispFeedsLA)),np.array(DispFeedsLA)[:,1],'k')
    ary,=plt.plot(range(len(DispFeedsRA)),np.array(DispFeedsRA)[:,1],'c')
    plt.legend([aly,ary],["aly","ary"])

    plt.subplot(10,1,5)
    #plt.axis([0,200,-1,1])
    blx,=plt.plot(range(len(DispFeedsLB)),np.array(DispFeedsLB)[:,0],'k')
    brx,=plt.plot(range(len(DispFeedsRB)),np.array(DispFeedsRB)[:,0],'c')
    plt.legend([blx,brx],["blx","brx"])

    
    plt.subplot(10,1,6)
    bly,=plt.plot(range(len(DispFeedsLB)),np.array(DispFeedsLB)[:,1],'k')
    bry,=plt.plot(range(len(DispFeedsRB)),np.array(DispFeedsRB)[:,1],'c')
    plt.legend([bly,bry],["bly","bry"])



    plt.subplot(10,1,7)
    #plt.axis([0,200,-1,1])
    clx,=plt.plot(range(len(DispFeedsLC)),np.array(DispFeedsLC)[:,0],'k')
    crx,=plt.plot(range(len(DispFeedsRC)),np.array(DispFeedsRC)[:,0],'c')
    plt.legend([clx,crx],["clx","crx"])


    
    plt.subplot(10,1,8)
    cly,=plt.plot(range(len(DispFeedsLC)),np.array(DispFeedsLC)[:,1],'k')
    cry,=plt.plot(range(len(DispFeedsRC)),np.array(DispFeedsRC)[:,1],'c')
    plt.legend([cly,cry],["cly","cry"])



    plt.subplot(10,1,9)
    #plt.axis([0,200,-1,1])
    dlx,=plt.plot(range(len(DispFeedsLD)),np.array(DispFeedsLD)[:,0],'k')
    drx,=plt.plot(range(len(DispFeedsRD)),np.array(DispFeedsRD)[:,0],'c')
    plt.legend([dlx,drx],["dlx","drx"])

    
    plt.subplot(10,1,10)
    dly,=plt.plot(range(len(DispFeedsLD)),np.array(DispFeedsLD)[:,1],'k')
    dry,=plt.plot(range(len(DispFeedsRD)),np.array(DispFeedsRD)[:,1],'c')
    plt.legend([dly,dry],["dly","dry"])

  

    #plt.legend([ed],["ED"])

    try:
      plt.show()
    except Exception as e:
      print e



    print "Enter category: ",
    category=raw_input()
    if category=="":
      category='c'

    elif "skip" in category:
      try:
        frames2skip=category.split(" ")[-1]
        frames2skip=int(frames2skip)
        continue
      except Exception as e:
        print e
        frames2skip=0
        

    '''

    features={#'category':category,
    'frame':FRAMES_ELAPSED,
    'BGSubFeat':BGSubFeeds,

    'dispLA':np.array(DispFeedsLA).tolist(),
    'dispLB':np.array(DispFeedsLB).tolist(),
    'dispLC':np.array(DispFeedsLC).tolist(),
    'dispLD':np.array(DispFeedsLD).tolist(),

    'dispRA':np.array(DispFeedsRA).tolist(),
    'dispRB':np.array(DispFeedsRB).tolist(),
    'dispRC':np.array(DispFeedsRC).tolist(),
    'dispRD':np.array(DispFeedsRD).tolist(),



    'distLA':np.array(DistFeedsLA).tolist(),
    'distLB':np.array(DistFeedsLB).tolist(),
    'distLC':np.array(DistFeedsLC).tolist(),
    'distLD':np.array(DistFeedsLD).tolist(),

    'distRA':np.array(DistFeedsRA).tolist(),
    'distRB':np.array(DistFeedsRB).tolist(),
    'distRC':np.array(DistFeedsRC).tolist(),
    'distRD':np.array(DistFeedsRD).tolist(),

    

    #,DispFeedsLB,DispFeedsLC,DispFeedsLD,DispFeedsRA,DispFeedsRB,DispFeedsRC,DispFeedsRD],
    #'dist':[DistFeedsLA,DistFeedsLB,DistFeedsLC,DistFeedsLD,DistFeedsRA,DistFeedsRB,DistFeedsRC,DistFeedsRD],


    'VcorrA':VcorrA.tolist(),
    'VcorrB':VcorrB.tolist(),
    'VcorrC':VcorrC.tolist(),
    'VcorrD':VcorrD.tolist(),
    'VcorrAB':VcorrAB.tolist(),
    'VcorrCD':VcorrCD.tolist(),
    'EcorrA':EcorrA.tolist(),
    'EcorrB':EcorrB.tolist(),
    'EcorrC':EcorrC.tolist(),
    'EcorrD':EcorrD.tolist(),

    'unVcorrA':unVcorrA.tolist(),
    'unVcorrB':unVcorrB.tolist(),
    'unVcorrC':unVcorrC.tolist(),
    'unVcorrD':unVcorrD.tolist(),
    'unVcorrAB':unVcorrAB.tolist(),
    'unVcorrCD':unVcorrCD.tolist(),

    'unEcorrA':unEcorrA.tolist(),
    'unEcorrB':unEcorrB.tolist(),
    'unEcorrC':unEcorrC.tolist(),
    'unEcorrD':unEcorrD.tolist()
    }


    featureDirectory = "extractedFeatures"

    if not os.path.exists(featureDirectory):
      os.mkdir(featureDirectory)

    
    with open("extractedFeatures/%s.txt" % camera_id.split("/")[-1],'a+') as f:
      f.write(json.dumps(features) + "\r\n")


    '''
    with open("extractedFeatures/%s.txt" % category,'a+') as f:
      f.write(json.dumps(features) + "\r\n")

    '''

    continue

    
with open("log.txt",'a+') as f:
  f.write(camera_id + "," + str(FRAMES_ELAPSED) +"\r\n")

print "END"
#raw_input()

cap.release()
cv2.destroyAllWindows()
