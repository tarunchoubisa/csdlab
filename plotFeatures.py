import matplotlib.pyplot as plt
import numpy as np
import sys,json



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
		f1.append(max(features['VcorrAB']))
		f2.append(max(features['VcorrCD']))
		#EcorrMax=max(np.array(features['EcorrA'])+np.array(features['EcorrB'])+np.array(features['EcorrC'])+np.array(features['EcorrD']))
		_f3=max([max(features['unVcorrA'])+max(features['unVcorrB']),max(features['unVcorrC'])+max(features['unVcorrD'])])
		f3.append(_f3)
		y.append(classy)

	f.close()





fill_data(sys.argv[1],0)
fill_data(sys.argv[2],1)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(f1,f2,f3,c=y, cmap=plt.cm.coolwarm)
plt.show()

#data=zip(f1,f2)

'''print data
import json
g=open("CvH_normalised_VAB_VCD.txt",'w')
g.write(json.dumps(data)+"\n")
g.write(json.dumps(y))
g.close()
'''



