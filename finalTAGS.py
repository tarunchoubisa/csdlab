import json,os,sys,time


finalTAGSfolder='fTAGS'

if not os.path.exists(finalTAGSfolder):
	os.mkdir(finalTAGSfolder)
	time.sleep(0.5)

try:
	file=sys.argv[1]
except:
	sys.exit(1)

f=open(file,'r')
data=f.read()
f.close()


data=json.loads(data)

lastclass=data[0][1]
lastindex=0

transitions=0

for i in range(len(data)):
	thisclass=data[i][1]
	if thisclass==lastclass:
		pass
	else:
		print lastclass,lastindex,"--->",thisclass,i,"diff:",i-lastindex
		lastindex=i
		lastclass=thisclass
		transitions+=1



print "-------------------------"
print "transitions",transitions



if transitions==2:
	ans='y'
	print "moving..."
else:
	ans='n'
	print "skipping..."

print "move ? y/n"
raw_input()

if ans=='y':
	os.system("mv " + file + " " +finalTAGSfolder)


