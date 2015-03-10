'''
#=============================================================================
#     FileName: calc_mw_logp.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-11 17:53:08
#   LastChange: 2015-03-10 16:23:18
#      History:
#=============================================================================
'''
import sys
from getopt import getopt
import pybel

def main(argv=sys.argv):
    if len(argv) not in (4,5):
        print "\n  Usage: %s [options] infile.list output"%argv[0]
        print "  [options]"
        print "  --logp: calculate logP"
        print "  --mw  : calculate MW"
        print "  at least one of `--logp` and `--mw` should be given"
        print ""
        sys.exit(1)

    candidates = []

    options,args = getopt(argv[1:],'',['logp','mw'])
    for opt,val in options:
        if opt == "--logp":
            candidates.append("logP")
        elif opt == "--mw":
            candidates.append("MW")
        else:
            print "Error: invalid option",opt
            sys.exit(1)

    assert len(args) == 2
    inf = open(args[0],'r')
    outf = open(args[1],'w')
    outf.write("name")
    if "logP" in candidates:
        outf.write("\tlogP")
    if "MW" in candidates:
        outf.write("\tMW")
    outf.write("\n")
    for line in inf:
        if line=="" or line.startswith("#"):
            continue
        line = line.strip()
        _format = line[line.rfind(".")+1:]
        mol = pybel.readfile(_format, line).next()
        desc = mol.calcdesc(candidates)
        outf.write(line)
        try:
            outf.write("\t%g"%desc["logP"])
        except:
            pass
        try:
            outf.write("\t%g"%desc["MW"])
        except:
            pass
        outf.write("\n")
    inf.close()
    outf.close()

main()

