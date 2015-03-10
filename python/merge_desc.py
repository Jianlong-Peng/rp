'''
#=============================================================================
#     FileName: merge_desc.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-11 18:04:24
#   LastChange: 2015-03-10 13:57:07
#      History:
#=============================================================================
'''
import sys

def main(argv=sys.argv):
    if len(argv) != 4:
        print "\n    OBJ: to add those in `global` to `local`"
        print "\n  Usage: %s local global output"%argv[0]
        print "  local : generate by calcDescriptors.py"
        print "  global: generate by calc_mw_logp.py"
        print "  output: "
        sys.exit(1)

    #load global
    inf = open(argv[2],'r')
    line = inf.readline()
    global_desc = {}
    for line in inf:
        line = line.split()
        global_desc[line[0]] = [line[1]]
    inf.close()

    inf = open(argv[1],'r')
    outf = open(argv[3],'w')
    line = inf.readline()
    while line != "":
        name = line.split()[0]
        outf.write(line)
        line = inf.readline()
        while line!="" and line.startswith("\t"):
            outf.write(line[:-1])
            for item in global_desc[name]:
                outf.write(",%s"%item)
            outf.write("\n")
            line = inf.readline()
    inf.close()
    outf.close()

main()

