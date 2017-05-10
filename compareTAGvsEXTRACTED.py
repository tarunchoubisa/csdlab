import sys,json,csv,os

file=sys.argv[1]

f=open(file,'r')
reader=csv.reader(f)


finalExtractedFeaturesFolder="fExtractedFeatures"

if not os.path.exists(finalExtractedFeaturesFolder):
	os.mkdir(finalExtractedFeaturesFolder)


for row in reader:
	 gnameRaw=row[0]
	 gname=row[0].split("/")[-1]
	 gname="fTAGS/"+gname+"_TAG.txt"
	 try:
	 	g=open(gname)
	 	gdata=json.loads(g.read())
	 	g.close()

	 	Nextracted=int(row[1])
	 	Ntagged=int(gdata[-1][0])
	 	print gname,Nextracted,Ntagged,Nextracted-Ntagged

	 	gnameRawfilename=gnameRaw.split("/")[-1]+".txt"
	 	os.rename('extractedFeatures/'+gnameRawfilename,finalExtractedFeaturesFolder+"/"+gnameRawfilename)
	 
	 except Exception as e:
	 	print e


f.close()