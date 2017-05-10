import os,sys,json,time


tagFolder="fTAGS"
featureFolder='fExtractedFeatures'

classfolder="classfolder"

if not os.path.exists(classfolder):
	os.mkdir(classfolder)
	time.sleep(0.25)


taglogfile=sys.argv[1]

f=open(taglogfile,'r')
taglog=f.read()
f.close()

taglog=taglog.split("\r\n")
taglog=taglog[:-1]

#print taglog

taglog=[json.loads(x) for x in taglog]


taglogFiltered=[]

for x in taglog:
	for y in x:
		if y['thisclass']=='c':
			taglogFiltered.append(y)




for tag in taglogFiltered:
	extractionFile=featureFolder+"/"+tag['file'][:-8]+".txt"
	videofilename=tag['file'][:-8]
	tagstart,tagend=int(tag['lastindex']),int(tag['thisindex'])
	tagclass=tag['lastclass']
	
	if not os.path.exists(extractionFile):
		print "file doesnt exist:",extractionFile
		raw_input()

	else:
		print "found file:",extractionFile,
		print tagclass,tagstart,tagend,tagend-tagstart,tag['diff']
		#print tagstart,tagstart<150
		#continue

		efile=open(extractionFile,'r')
		for newline in efile:
			newfeature=json.loads(newline.strip())
			newfeature['file']=videofilename

			endBuffFeatures = int(newfeature['frame'])
			
			lengthBuffFeatures=len(newfeature['BGSubFeat'])

			startBuffFeatures = endBuffFeatures - lengthBuffFeatures
			
			if startBuffFeatures<=0:
				startBuffFeatures=0
			

			# |  BuffFeature |
			# +  tag  +


			# |  ++++  |
 			if startBuffFeatures<tagstart and tagend<endBuffFeatures:
				#print startBuffFeatures,tagstart,tagend,endBuffFeatures
				#print tagclass
				classfilename=tagclass
				#print "class:"+classfilename
				


			
			# |      |   ++++
			elif endBuffFeatures<tagstart:
				classfilename='c'
			
			

			# ++++  |       |
			elif tagend<startBuffFeatures:
				classfilename='c'


			else:
				classfilename='ambiguous'


			with open(classfolder+"/"+classfilename+".txt",'a+') as cf:
				cf.write(json.dumps(newfeature) + "\r\n")

