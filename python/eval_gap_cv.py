'''
#=============================================================================
#     FileName: eval_gap_cv.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Version: 0.0.1
#   LastChange: 2014-12-29 06:06:59
#      History:
#=============================================================================
'''
import sys
from getopt import getopt
import numpy as np

def main(argv=sys.argv):
	if len(argv) < 2:
		print """
OBJ
  to calculate mean and stdev for each column

Usage
  %s [options] input

Options
  --row-name : <default: False>
               to specify `input` has row name in 1st column
  --col-name : <default: False>
               to specify `input` has col name in {N+1}st row
  --ignore N : <default: 0>
               to specify number of lines being ignored
"""%argv[0]
		sys.exit(1)
	
	has_row_name = False
	has_col_name = False
	N = 0

	options,args = getopt(argv[1:],"",["row-name","col-name","ignore="])
	for opt,val in options:
		if opt == "--row-name":
			has_row_name = True
		elif opt == "--col-name":
			has_col_name = True
		elif opt == "--ignore":
			N = int(val)
		else:
			print "Error: invalid option",opt
			sys.exit(1)
	assert len(args) == 1
	
	inf = open(args[0],'r')
	for i in xrange(N):
		line = inf.readline()
	dataset = []
	title = []
	if has_col_name:
		if has_row_name:
			title = inf.readline().split()[1:]
		else:
			title = inf.readline().split()
	line = inf.readline()
	while line != "":
		if has_row_name:
			dataset.append(map(float,line.split()[1:]))
		else:
			dataset.append(map(float,line.split()))
		line = inf.readline()
	inf.close()
	dataset = np.asarray(dataset)

	avers = np.mean(dataset, axis=0)
	stdev = np.std(dataset, axis=0, ddof=1)

	print "         mean     stdev"
	for i in xrange(avers.shape[0]):
		if title:
			print "%-8s %-8.6f %-.6f"%(title[i], avers[i], stdev[i])
		else:
			print "#%-7d %-8.6f %-.6f"%(i+1, avers[i], stder[v])

main()

