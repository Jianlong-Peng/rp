'''
#=============================================================================
#     FileName: addYValues.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2013-09-24 14:50:39
#   LastChange: 2013-09-24 15:01:44
#      History:
#=============================================================================
'''

import sys
import os

def main(argv=sys.argv):
    if len(argv) != 4:
        print "\n  Usage: %s in.txt yvalues out.txt"%argv[0]
        print "  in.txt : generated via `calcDescriptor.py`"
        print "  yvalues: it must has column `No.` and `log(CLint)`"
        print "  out.txt: where to save the dataset\n"
        sys.exit(1)

    inf = open(argv[2],"r")
    line = inf.readline()
    line = line.strip().split("\t")
    no_index = line.index("No.")
    y_index = line.index("log(CLint)")
    database = {}  #{no:y-value, ...}
    for line in inf:
        line = line.strip().split("\t")
        database[line[no_index]] = line[y_index]
    inf.close()

    inf = open(argv[1],"r")
    outf = open(argv[3],"w")
    line = inf.readline()
    while line != "":
        line = line.strip()
        root,ext = os.path.splitext(os.path.basename(line))
        if not database.has_key(root):
            print >>sys.stderr, "Can't find y-value for %s"%line
            print >>outf, line
        else:
            print >>outf, "%s\t%s"%(line,database[root])
        line = inf.readline()
        while line.startswith("\t"):
            outf.write(line)
            line = inf.readline()
    inf.close()
    outf.close()

main()


