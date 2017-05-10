import sys,json,os,time


filename=sys.argv[1]

TAGSverifiedFolder="vTAGS"

if not os.path.exists(TAGSverifiedFolder):
	print "creating dir..",TAGSverifiedFolder
	os.mkdir(TAGSverifiedFolder)
	time.sleep(1)
else:
	print "dir exists...",TAGSverifiedFolder





f=open(filename,'r')

data=f.read()
f.close()


data=json.loads(data)

categories=[]

for tag in data:
	if tag[1] in categories:
		pass
	else:
		categories.append(tag[1])

for i in range()


print categories,len(categories)


if len(categories)<2:
	print "category contains less than 2 values"
	raw_input()
	sys.exit(0)

if len(categories)>2:
	print "Skipping move..."
else:
	print "moving..."
	os.system("mv " + filename + " " + TAGSverifiedFolder)


#time.sleep(2)