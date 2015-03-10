'''
#=============================================================================
#     FileName: calc_mw_logp.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-11 17:53:08
#   LastChange: 2014-10-11 17:58:10
#      History:
#=============================================================================
'''
import sys
import pybel

def main(argv=sys.argv):
    if len(argv) != 3:
        print "\n  Usage: %s infile.list output"%argv[0]
        print "  to calculate logP and MW"
        print ""
        sys.exit(1)

    inf = open(argv[1],'r')
    outf = open(argv[2],'w')
    print >>outf, "name\tlogP\tMW"
    for line in inf:
        if line=="" or line.startswith("#"):
            continue
        line = line.strip()
        _format = line[line.rfind(".")+1:]
        mol = pybel.readfile(_format, line).next()
        desc = mol.calcdesc(["logP","MW"])
        print >>outf, "%s\t%g\t%g"%(line, desc["logP"], desc["MW"])
    inf.close()
    outf.close()

main()

