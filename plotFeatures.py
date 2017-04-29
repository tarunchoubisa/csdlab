import matplotlib.pyplot as plt
import numpy as np
import sys,json

feature_file=sys.argv[1]
f=open(feature_file,'r')

f1=[]#[10,0]
f2=[]#[0,10]
y=[]#[0,1]

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
	y.append(0)


feature_file=sys.argv[2]
f=open(feature_file,'r')


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
	y.append(1)

#plt.axis([0,50000,0,50000])
plt.scatter(f1,f2,c=y, cmap=plt.cm.coolwarm)
plt.show()

data=zip(f1,f2)

print data
import json
g=open("CvH_normalised_VAB_VCD.txt",'w')
g.write(json.dumps(data)+"\n")
g.write(json.dumps(y))
g.close()




