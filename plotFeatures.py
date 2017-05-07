import matplotlib.pyplot as plt
import numpy as np
import sys,json



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


f1=[]#[10,0]
f2=[]#[0,10]
f3=[]
y=[]#[0,1]

def fill_data(feat_file,classy):
	feature_file=sys.argv[1]
	f=open(feat_file,'r')
	while 1:
		line=f.readline().strip()
		if not line:break
		features=json.loads(line)
		#print features

		#f1.append(max(features['VcorrAB']))
		#f2.append(max(features['VcorrCD']))
		#f1.append(max(features['unEcorrA'])+max(features['unEcorrB']))
		#f2.append(max(features['unEcorrC'])+max(features['unEcorrD']))
		#abcorr=np.correlate(features['unVcorrAB'],features['unVcorrAB'])
		#cdcorr=np.correlate(features['unVcorrCD'],features['unVcorrCD'])
		#f1.append(max(features['VcorrAB']))
		#f2.append(max(features['VcorrCD']))
		#f1.append(0)

		f1.append(max(features['VcorrCD']))
		f2.append(max(features['VcorrAB']))

		#f2.append(0)
		#EcorrMax=max(np.array(features['EcorrA'])+np.array(features['EcorrB'])+np.array(features['EcorrC'])+np.array(features['EcorrD']))
		#_f3=max([max(features['unVcorrA'])+max(features['unVcorrB']),max(features['unVcorrC'])+max(features['unVcorrD'])])
		#f3.append(_f3)

		DispFeedsLBC=np.append(np.array(features['dispLB']),np.array(features['dispLC']),axis=1)
		DispFeedsRBC=np.append(np.array(features['dispRB']),np.array(features['dispRC']),axis=1)

		f3.append(max(VectorCorr(DispFeedsLBC,DispFeedsRBC,normalise=1)))
		y.append(classy)

	f.close()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fill_data(sys.argv[1],0)
fill_data(sys.argv[2],1)



if len(sys.argv)>2:
	fill_data(sys.argv[3],2)


if "3D" in "-".join(sys.argv):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(f1,f2,f3,c=y, cmap=plt.cm.cool)
	#plt.xlim(-100,100)
else:
	plt.scatter(f1,f2,c=y, cmap=plt.cm.cool)

plt.show()

#data=zip(f1,f2)

'''print data
import json
g=open("CvH_normalised_VAB_VCD.txt",'w')
g.write(json.dumps(data)+"\n")
g.write(json.dumps(y))
g.close()
'''



