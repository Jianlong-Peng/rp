'''
#=============================================================================
#     FileName: scale.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2013-09-23 20:43:19
#   LastChange: 2014-04-09 10:49:21
#      History:
#=============================================================================
'''
import sys
from getopt import getopt
from copy import deepcopy

def generate_parameter(infile):
    '''
    parameter
    =========
    infile: string, generated by `calcDescriptor.py`

    return
    ======
    para: dict
      {atom_type:[[min...],[max...]], ...}
    '''
    descriptors = {}   #{atom_type:[[values...],[values...],...], ...}
    inf = open(infile,"r")
    line = inf.readline()
    while line != "":
        line = inf.readline()
        while line.startswith("\t"):
            line = line.strip().split(":")
            atom_type = line[1]
            values = map(float,line[2].split(","))
            if not descriptors.has_key(atom_type):
                descriptors[atom_type] = []
            descriptors[atom_type].append(values)
            line = inf.readline()
    inf.close()
    para = {}   #{atom_type:[[min...],[max...]],...}
    for atom_type in descriptors.iterkeys():
        para[atom_type] = [deepcopy(descriptors[atom_type][0]),deepcopy(descriptors[atom_type][0])]
        for i in xrange(1,len(descriptors[atom_type])):
            for j in xrange(len(descriptors[atom_type][i])):
                if descriptors[atom_type][i][j] < para[atom_type][0][j]:
                    para[atom_type][0][j] = descriptors[atom_type][i][j]
                if descriptors[atom_type][i][j] > para[atom_type][1][j]:
                    para[atom_type][1][j] = descriptors[atom_type][i][j]
    return para

def save_parameter(para,outfile):
    '''
    parameter
    =========
    para: dict, {atom_type:[[min...],[max...]], ...}
    outfile: string, where to save parameters
        parameters will be saved as follows:
        atom_type
        \\tmin max
        \\t...
        atom_type
        \\tmin max
        \\t...
        ...
    '''
    outf = open(outfile,"w")
    for key in para.iterkeys():
        print >>outf, key
        for i in xrange(len(para[key][0])):
            print >>outf, "\t%.16g %.16g"%(para[key][0][i],para[key][1][i])
    outf.close()

def read_parameter(infile):
    '''
    to read scaling parameters from `infile`
    '''
    para = {}  # {atom_type:[[min...],[max...]],...}
    inf = open(infile,"r")
    line = inf.readline()
    while line != "":
        atom_type = line.strip()
        if para.has_key(atom_type):
            print >>sys.stderr, "Error: more than one set of scalling parameters found for atom type",atom_type
            inf.close()
            sys.exit(1)
        para[atom_type] = [[],[]]
        line = inf.readline()
        while line.startswith("\t"):
            line = line.split()
            para[atom_type][0].append(float(line[0]))
            para[atom_type][1].append(float(line[1]))
            line = inf.readline()
    inf.close()
    return para

def scale(orig_value, min_, max_):
    if min_ == max_:
        #return orig_value
        return 0.
    else:
        return 1.*(orig_value-min_)/(max_-min_)

def runScale(para, infile, outfile, verbose):
    '''
    to scale `infile` according to para, and scaled
    values will be saved in `outfile`
    '''
    inf = open(infile,"r")
    outf = open(outfile,"w")
    line = inf.readline()
    while line != "":
        outf.write(line)
        name = line.split()[0]
        line = inf.readline()
        while line.startswith("\t"):
            line = line.strip().split(":")
            if not para.has_key(line[1]):
                print >>sys.stderr,"Error: Can't find scalling parameters for atom type",line[1]
                inf.close()
                outf.close()
                sys.exit(1)
            min_max = para[line[1]]
            orig_values = line[2].split(",")
            if len(min_max[0]) != len(orig_values):
                print >>sys.stderr, "Error: different number of descriptors found for atom type",line[1]
                inf.close()
                outf.close()
                sys.exit(1)
            scaled_value = scale(float(orig_values[0]),min_max[0][0],min_max[1][0])
            if verbose and (scaled_value<=-0.5 or scaled_value>=1.5):
                print "Warning:",name,line[0],line[1],"1",scaled_value
            outf.write("\t%s:%s:%.6g"%(line[0],line[1],scaled_value))
            for i in xrange(1,len(orig_values)):
                scaled_value = scale(float(orig_values[i]),min_max[0][i],min_max[1][i])
                if verbose and (scaled_value<=-0.5 or scaled_value>=1.5):
                    print "Warning:",name,line[0],line[1],i+1,scaled_value
                outf.write(",%.6g"%scaled_value)
            outf.write("\n")
            line = inf.readline()
    inf.close()
    outf.close()

def main(argv=sys.argv):
    if len(argv)!=5 and len(argv)!=6:
        print "\nUsage:"
        print "  %s [options] infile outfile"%argv[0]
        print "\nOptions:"
        print "  -s save_filename: save scaling parameters"
        print "  -r restore_filename: restore scaling parameters"
        print "  --verbose: if given, display those with scaled value <=-0.5 or >=1.5"
        print "\nAttention:"
        print "  . if `-s` is given, `infile` will be scalled to (-1,1),"
        print "    and parameters will be saved in `save_filename`"
        print "  . if `-r` is given, scaling `infile` using `restore_filename` instead."
        print ""
        sys.exit(1)
    
    options,args = getopt(argv[1:],"s:r:",["verbose"])
    if len(args) != 2:
        print "Error: invalid number of arguments"
        sys.exit(1)

    save_file = None
    load_file = None
    verbose = False

    for opt,value in options:
        if opt == "-s":
            save_file = value
        elif opt == "-r":
            load_file = value
        elif opt == "--verbose":
            verbose = True
        else:
            print "Error: invalid option ",opt
            sys.exit(1)

    if save_file is not None:
        para = generate_parameter(args[0])
        save_parameter(para,save_file)
    if load_file is not None:
        para = read_parameter(load_file)
    runScale(para,args[0],args[1],verbose)

main()

