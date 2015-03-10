'''
#=============================================================================
# FileName: tools.py
# Desc: 
#   Author: jlpeng
#Email: jlpeng1201@gmail.com
# HomePage: 
#  Version: 0.0.1
#   LastChange: 2015-03-10 09:45:54
#  History:
#=============================================================================
'''

def report(actual, predict, k):
	results = []
	for i in xrange(1,k+1):
		total,miss,right = estimate(actual, predict, i)
		results.append((i,right,total-miss-right))
	print "totally %d samples, of which %d has no SOM labeled"%(total,miss)
	print "k  right error accuracy"
	for i,right,error in results:
		print "%-2d %-5d %-5d %-g"%(i,right,error,1.*right/(right+error))
	print ""


def load_predict(infile):
	predict = {}
	inf = open(infile,'r')
	for line in inf:
		line = line.split()
		if len(line) == 2:
			continue
		name = line[0].split("\\")[-1].split(".")[0]
		values = []
		if ":" in line[2]:
			i = 2
		else:
			i = 3
		for j in xrange(i,len(line)):
			idx,val = line[j].split(":")
			values.append((idx,float(val)))
		values.sort(key=lambda x:x[1], reverse=True)
		predict[name] = values
	inf.close()
	return predict


def load_des(infile,candidates=[]):
	des = {}
	inf = open(infile,'r')
	line = inf.readline()
	while line != "":
		name = line.split()[0].split("\\")[-1]
		name = name[:name.rfind(".")]
		if candidates and name not in candidates:
			line = inf.readline()
			continue
		des[name] = []
		line = inf.readline()
		while line!="" and line.startswith("\t"):
			temp = line.strip().split(",")[0]
			atom,type,val = temp.split(":")
			des[name].append((atom,type))
			line = inf.readline()
	inf.close()
	return des

def estimate(actual, predict, k):
	total = 0
	miss  = 0    #there is no SOM labeled
	right = 0
	for name in actual.keys():
		total += 1
		if len(actual[name]) == 0:
			miss += 1
			continue
		pred_values = predict[name]
		found = False
		for site,y in pred_values[:k]:
			if site in actual[name]:
				found = True
		if found:
			right += 1
	return total,miss,right

