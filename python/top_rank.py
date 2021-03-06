'''
#=============================================================================
#     FileName: top_rank.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Version: 0.0.1
#   LastChange: 2015-03-10 02:13:02
#      History:
#=============================================================================
'''
import sys
from tools import report, load_des, load_predict

def main(argv=sys.argv):
	if len(argv) != 5:
		print """
OBJ
  to evaluate top-k accuracy

Usage
  %s k predict som

Arguments
  k      : int, estimate top-1 to top-K accuracy
  predict: file, generated by gap_predict
  som    : file, som file.
  des    : file, descriptors, same as input for gap_predict
 
Attention
  1. reports are based on SOMs involving only one atom:
     - considering all types of SOMs
	 - exclude SOM type `6`(O-conjugation)
"""%argv[0]
		sys.exit(1)
	
	des = load_des(argv[4])   #key=mol_name, value=[(atom,type),...]
	k = int(argv[1])
	predict = load_predict(argv[2])  #key=name, value=[[site,y],...] which has been sorted
	actual_all,actual_no6 = load_som(argv[3],des)      #key=name, value=[site1,site2,...]

	print "===report considering all types of SOMs except those with more than one atoms==="
	report(actual_all, predict, k)
	print "\n===report excluding SOM type 6 (O-conjugation) and more than one atoms==="
	report(actual_no6, predict, k)


def valid(actual,atom):
	for a,t in actual:
		if a==atom and t=='6':
			return False
	return True

def load_som(infile,des):
	actual_all = {}
	actual_no6 = {}
	inf = open(infile,'r')
	line = inf.readline()
	for line in inf:
		line = line.strip().split("\t")
		actual_all[line[0]] = []
		actual_no6[line[0]] = []
		for atom in line[3:]:
			if "-" in atom:
				continue
			actual_all[line[0]].append(atom)
			if valid(des[line[0]],atom):
				actual_no6[line[0]].append(atom)
	inf.close()
	return actual_all,actual_no6


main()



