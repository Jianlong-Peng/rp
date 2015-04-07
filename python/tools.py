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
        total,miss,right,fail = estimate(actual, predict, i)
        results.append((i,right,total-miss-right-fail))
    print "totally %d samples, of which %d has no SOM labeled, %d can't be predicted"%(total,miss,fail)
    print "k  right error accuracy"
    for i,right,error in results:
        print "%-2d %-5d %-5d %-g"%(i,right,error,1.*right/(right+error))
    print ""

    iap = calcIAP(actual, predict)
    print "IAP=%g\n"%iap


def valid(actual, atom):
    for a,t in actual:
        if a==atom and t=='6':
            return False
    return True

def load_predict(infile,des):
    predict_all = {}
    predict_no6 = {}
    inf = open(infile,'r')
    for line in inf:
        line = line.split()
        if len(line) == 2:
            continue
        name = line[0].split("\\")[-1].split(".")[0]
        values_all = []
        values_no6 = []
        if ":" in line[2]:
            i = 2
        else:
            i = 3
        for j in xrange(i,len(line)):
            idx,val = line[j].split(":")
            values_all.append((idx,float(val)))
            if valid(des[name], idx):
                values_no6.append((idx,float(val)))
        values_all.sort(key=lambda x:x[1], reverse=True)
        values_no6.sort(key=lambda x:x[1], reverse=True)
        predict_all[name] = values_all
        predict_no6[name] = values_no6
    inf.close()
    return predict_all,predict_no6


def load_des(infile,candidates=[],delta=None):
    des = {}
    inf = open(infile,'r')
    line = inf.readline()
    while line != "":
        name = line.split()[0].split("\\")[-1]
        name = name[:name.rfind(".")]
        if candidates and name not in candidates:
            line = inf.readline()
            continue
        values = []  #[(atom,type),...]
        bad = False
        line = inf.readline()
        while line!="" and line.startswith("\t"):
            temp = line.strip().split(",")
            atom,type,val = temp[0].split(":")
            values.append((atom,type))
            for _ in map(float,temp[1:]):
                if (delta is not None) and (_<(-1.*delta) or _>(1+delta)):
                    bad = True
                    break
            line = inf.readline()
        if not bad:
            des[name] = values
    inf.close()
    return des

def estimate(actual, predict, k):
    total = 0
    miss  = 0    #there is no SOM labeled
    right = 0
    fail  = 0    #number of samples can't be predicted.
    for name in actual.keys():
        total += 1
        if not predict.has_key(name):
            fail += 1
            continue
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
    return total,miss,right,fail

def calcIAP(actual, predict):
    pos = []
    neg = []
    for name in predict.keys():
        for atom,y in predict[name]:
	    if atom in actual[name]:
	        pos.append(y)
	    else:
	        neg.append(y)
    num_large = 0
    for val1 in pos:
        for val2 in neg:
	    if val1 > val2:
	        num_large += 1
    ret = 1.0 * num_large / (len(pos)*len(neg))
    return ret

