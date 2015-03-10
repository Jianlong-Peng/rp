'''
#=============================================================================
#     FileName: calcDescriptor.py
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2013-09-22 09:23:05
#   LastChange: 2015-03-10 16:01:51
#      History:
#=============================================================================
'''
import sys
import os
import pybel
from getopt import getopt,GetoptError
#from tools import matchAtoms

description = """
===========DESCRIPTION ABOUT DESCRIPTORS TO BE CALCULATED===========
For all potential sites, some quantum-chemical descriptors will be
calculated. Here, we will use pre-defined SMARTS strings to match
those candidate atoms, and parse Mopac output file to generate
corresponding descriptors. The belowing are some more detailed explanation.

The atom types being concerned:
(the number at the begining of each line will be)
(used as indicator of that atom type.)
0. N-dealkylation                   [$([C^3X4;!H0][#7])]
1. O-dealkylation                   [$([C^3X4;!H0][#8])]
2. benzylic/allylic C-hydroxylation [$([C^3X4;!H0][c,$(C=*),$(C#*)])]
3. aliphatic C-hydroxylation        [C^3X4;!H0]
4. aromatic c-hydroxylation         [cH]
5. N-Conjugation/oxydation          [$([#7;R]),$([NX3]);!$([#7][C,S]=[O,S,N]);!$(N=O)]
6. O-Conjugation                    [OX2H]

Descriptors to be calculated:
- one-center terms (8)
  RA     : Fukui one-electron reactivity index
  NA     : Fukui nucleophilic reactivity index
  EA     : Fukui electrophilic reactivity index
  SEA    : electrophilic superdelocalizability
  SNA    : nucleophilic superdelocalizability
  atomicQ: atomic charge (Mulliken)
  selfQ  : the charge kept by the atoms involved in the bond.
  activeQ: the charge not involved in bonding.
- two-center terms
  Bsigpi : sigma-pi bond order
  B      : bond order (Mulliken)
  E2     : two-center total of electronic and nuclear energies
  polarAB: polarizability
- topological (1)   if --span given
  SPAN   : max shortest distance from candidate site to other atoms
           divided by max shortest distance between any atoms
- logP (1)   if --logp given

As those two-center terms involve two atoms, the following strategies
will be applied to generate atomic descriptors
- for atom types 0, 1, 2, 3:
  two-center terms for all C-H bonds, followed by max,min,mean,sum.
- for atom type 4:
  two-center terms for c-H bond only.
- for atom type 5:
  two-center terms for all connected bonds, followed by max,min,mean,sum
- for atom type 6:
  to calculate two-center terms for O-H and O-C bond.

Thus, the following descriptors will be calculated
(values of descriptors will be saved in the same order as listed below)
- for atom type 0,1,2,3  (8+5*4(+1+1) descriptors)
  RA,NA,EA,SEA,SNA,atomicQ,selfQ,activeQ,
  Bsigsig_max(min,mean,sum), Bsigpi_max(min,mean,sum)
  B_max(min,mean,sum), E2_max(min,mean,sum)
  polarAB_max(min,mean,sum)
  SPAN
  logp
- for atom type 4          (8+5*1(+1+1) descriptors)
  RA,NA,EA,SEA,SNA,atomicQ,selfQ,activeQ,
  B_cH, E2_cH, Bsigsig_cH, Bsigpi_cH, polarAB_cH
  SPAN
  logP
- for atom type 5  (8+6*4(+1+1))
  RA,NA,EA,SEA,SNA,atomicQ,selfQ,activeQ,
  Bsigsig_max(min,mean,sum), Bsigpi_max(min,mean,sum), Bpipi_max(min,mean,sum)
  B_max(min,mean,sum), E2_max(min,mean,sum)
  polarAB_max(min,mean,sum)
  SPAN
  logP
- for atom type 6          (8+5+3*2(+1+1) descriptors)
  RA,NA,EA,SEA,SNA,atomicQ,selfQ,activeQ,
  Bsigsig_OH, Bsigsig_OX, Bsigpi_OH, Bsigpi_OX, Bpipi_OX
  B_OH,B_OX,E2_OH,E2_OX,polarAB_OH,polarAB_OX
  SPAN
  logP

finally, calculated descriptors will be saved as follows:
    name id:type:d1,d2,...,dn id:type:d1,d2,...,dn ...
where, `name` is the molecular name, `id` is the atomic ID,
`type` is the corresponding atom type, `di`s are values of descriptors.
===========================================================================
                                                       jlpeng1201@gmail.com
"""

global DEBUG
DEBUG = False

global do_span
do_span = False
global do_logp
do_logp = False

#called by 'calcDescriptor'
#only match those atoms in `candidate_atoms` if `candidate_atoms` is not None
#atom sites:
#  1. N-dealkylation                       [C^3X4;!H0][#7]
#  2. O-dealkylation                       [C^3X4;!H0][#8]
#  3. benzylic/allylic C-hydroxylation     [C^3X4;!H0]([#6,#1])([#6,#1])([#6,#1])[a,$(C=*),$(C#*)]
#  4. aliphatic C-hydroxylation            [C^3X4;!H0]([#6,#1])([#6,#1])([#6,#1])[#6,#1]
#  5. aromatic C-hydroxylation             [cH]
#  6. N-Conjugation/oxydation              [$([#7;R]),$([NX3]);!$([#7][C,S]=[O,S,N]);!$(N=O)]
#  7. O-Conjugation                        [OX2H]
#
def matchAtoms(mol, candidate_atoms):
    matches = [[] for _ in xrange(7)]
    for atom in mol.atoms:
        if (candidate_atoms is not None) and (atom.idx not in candidate_atoms):
            continue
        index = -1
        if atom.OBAtom.MatchesSMARTS("[C^3X4;!H0][#7]"):
            index = 0
        elif atom.OBAtom.MatchesSMARTS("[C^3X4;!H0][#8]"):
            index = 1
        elif atom.OBAtom.MatchesSMARTS("[C^3X4;!H0][!#6;!#1]"):
            index = -1
        elif atom.OBAtom.MatchesSMARTS("[C^3X4;!H0][c,$(C=*),$(C#*)]"):
            index = 2
        elif atom.OBAtom.MatchesSMARTS("[C^3X4;!H0]"):
            index = 3
        elif atom.OBAtom.MatchesSMARTS("[cH]"):
            index = 4
        elif atom.OBAtom.MatchesSMARTS("[$([#7;R]),$([NX3]);!$([#7][C,S]=[O,S,N]);!$(N=O)]"):
            index = 5
        elif atom.OBAtom.MatchesSMARTS("[OX2H]"):
            index = 6
        else:
            index = -1
        if index != -1:
            matches[index].append(atom.idx)

    return matches

#called by 'calcDescriptor'
def calcOneCenter(inf,descriptor1,homo_index):
    '''
    to calculate those one-center terms

    parameters
    ==========
    infile: string, .mopout file
    descriptor1: dict, {atomid:[atomtype],...}
      after this function being called, `descriptor1` will
      be updated. {atomid:[atomtype,descriptors...],...}
    homo_index: int. begin with 0.
    '''
    assert descriptor1 != {}
    line = inf.readline()
    while line!="" and line.strip()!="EIGENVECTORS":
        line = inf.readline()
    if line == "":
        print "Error: can't find `EIGENVECTOR`"
        inf.close()
        sys.exit(1)
    eigenvector = []  #[{atom:[AO...],...},...]  len(eigenvector) = num. of Mols
    mo_energy = []    #len(mo_energy) = num. of MOs
    line = inf.readline()  #empty line
    line = inf.readline()  #empty line
    line = inf.readline()  #Root No. ...
    begin_i = 0;
    while "Root No." in line:
        temp = line.split()
        no_newMO = len(temp)-2
        for i in xrange(no_newMO):
            eigenvector.append({})
        line = inf.readline()  #empty line
        line = inf.readline()  #1 A  2 A ...
        line = inf.readline()  #empty line
        line = inf.readline()  #energies
        mo_energy.extend(map(float,line.split()))
        line = inf.readline()  #empty line
        line = inf.readline()  #begin coefficient block
        previous_id = 0
        while line.strip() != "":
            current_id = int(line[7:11])
            if current_id != previous_id:
                for i in xrange(no_newMO):
                    eigenvector[begin_i+i][current_id] = []
                previous_id = current_id
            coeff = line[11:].split()
            for i in xrange(len(coeff)):
                eigenvector[begin_i+i][current_id].append(float(coeff[i]))
            line = inf.readline()
        begin_i += no_newMO
        line = inf.readline()  #empty line
        line = inf.readline()  #maybe "  Root No. ..."
    #end of getting orbitals
    for key in descriptor1.iterkeys():
        #1. RA
        ra = 0.;
        for j in xrange(len(eigenvector[homo_index][key])):
            ra += (eigenvector[homo_index][key][j]*eigenvector[homo_index+1][key][j])
        ra /= (mo_energy[homo_index+1]-mo_energy[homo_index])
        descriptor1[key].append(ra)
        #2. NA
        na = sum(map(lambda x:pow(x,2),eigenvector[homo_index][key])) / (1.-mo_energy[homo_index])
        descriptor1[key].append(na)
        #3. EA
        ea = sum(map(lambda x:pow(x,2),eigenvector[homo_index+1][key])) / (10.+mo_energy[homo_index+1])
        descriptor1[key].append(ea)
        #4. SEA
        sea = 0.
        for j in xrange(homo_index+1):
            if mo_energy[j] == 0.:
                continue
            sea += (sum(map(lambda x:pow(x,2),eigenvector[j][key]))/mo_energy[j])
        sea *= 2.
        descriptor1[key].append(sea)
        #5. SNA
        sna = 0.
        for j in xrange(homo_index+1,len(mo_energy)):
            if mo_energy[j] == 0.:
                continue
            sna += (sum(map(lambda x:pow(x,2),eigenvector[j][key]))/mo_energy[j])
        sna *= 2.
        descriptor1[key].append(sna)
    #6. atomicQ
    if line.strip() != "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS":
        print "Error: can't find `NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS`"
        inf.close()
        sys.exit()
    line = inf.readline()  #empty line
    line = inf.readline()  #ATOM NO.  TYPE  CHARGE  ATOM ELECTRON DENSITY
    line = inf.readline()
    while "DIPOLE" not in line:
        line = line.split()
        atom_id = int(line[0])
        if descriptor1.has_key(atom_id):
            descriptor1[atom_id].append(float(line[2]))
        line = inf.readline()
    #7. selfQ, activeQ
    line = inf.readline()
    while line!="" and ("SELF-Q" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find `SELF-Q`"
        inf.close()
        sys.exit(1)
    line = inf.readline()  #empty line
    line = inf.readline()  #empty line
    line = inf.readline()
    atom_id = 0
    while line!="" and ("CLOSED SHELL" not in line):
        line = line.split()
        atom_id += 1
        if descriptor1.has_key(atom_id):
            #selfQ
            descriptor1[atom_id].append(float(line[1]))
            #activeQ
            descriptor1[atom_id].append(float(line[2]))
        line = inf.readline() #empty line
        line = inf.readline()
    if line == "":
        print "There may be something wrong when generating SELF-Q and ACTIVE-Q"
        inf.close()
        sys.exit(1)
    return eigenvector,mo_energy

#called by 'calcTwoCenter'
#2013-09-24 21:31   do not compute those features!!!!!!!!!!!!!!!!!!!!!!
#for [cH]: Bsigsig_cH, Bsigpi_cH
#for [C]: Bsigsig_OH,Bsigpi_OH, followed by max,min,mean,sum
#for [N]: Bsigsig,Bsigpi,Bpipi, followed by max,min,mean,sum
#for [O]: Bsigsig_OH,Bsigsig_OX,Bsigpi_OH,Bsigpi_OX,Bpipi_OX
def sigmaPiBond(inf,descriptor,atom_nbors,atom_symbol):
    natom = len(atom_symbol)
    #can be furthur optimized such that it only records those pairs in `atom_nbors`!!!!!!!!!
    bond_matrix = [[] for _ in xrange(natom)] #lower triangle matrix
    for i in xrange(len(bond_matrix)):
        for j in xrange(0,i+1):
            bond_matrix[i].append([0.]*9)
    line = inf.readline()
    while line!="" and ("SIGMA-PI BOND-ORDER MATRIX" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find \"SIGMA-PI BOND-ORDER MATRIX\""
        inf.close()
        sys.exit(1)
    #to read SIGMA-PI BOND ORDER MATRIX
    index_ = {"S-SIGMA":0,"P-SIGMA":1,"P-PI":2}
    atoms_read = 0  # atoms read in the column
    line = inf.readline() #empty line
    line = inf.readline() #  S-SIGMA  P-SIGMA ...
    while ("SIGMA" in line) or ("PI" in line):
        y_types = line.split() #['S-SIGMA','P-SIGMA',...]
        y_atoms = []
        for i in xrange(len(y_types)):
            if y_types[i] == "S-SIGMA":
                atoms_read += 1
            y_atoms.append(atoms_read)
        line = inf.readline()  # C  1   C  1...
        line = inf.readline()  # ------------
        line = inf.readline()  # suppose to be "S-SIGMA  C  1   0.923585"
        search_atom = 0
        while search_atom < natom:
            temp = line.split()
            if len(temp) == 1:
                line = inf.readline()
            if line.strip() == "":
                for i in xrange(4):
                    line = inf.readline()
            x_type = line[:8].strip()
            search_atom = int(line[11:15])
            orders = line[15:].split()
            for m in xrange(len(orders)):
                k = index_[x_type]*3+index_[y_types[m]]
                bond_matrix[search_atom-1][y_atoms[m]-1][k] = float(orders[m])
                if search_atom==y_atoms[m] and (k not in (0,4,8)):
                    k = index_[y_types[m]]*3+index_[x_type]
                    bond_matrix[search_atom-1][y_atoms[m]-1][k] = float(orders[m])
            line = inf.readline()
        while line.strip() != "":
            line = inf.readline()
        line = inf.readline()  #next block
    #end of 'while ("SIGMA" in line) or ("PI" in line):'
    #go to calculate descriptors (Bsigsig, Bsigpi, Bpipi)
    for atom_id in descriptor.iterkeys():
        nbors_id = atom_nbors[atom_id]
        #[cH], Bsigsig and Bsigpi for c-H  (2)
        #for atom type 4
        if descriptor[atom_id][0] == 4:
            i = atom_id-1; j=nbors_id[0]-1
            if nbors_id[0] > atom_id:
                i,j = j,i
            bond_order = bond_matrix[i][j]
            #1. Bsigsig
            descriptor[atom_id].append(bond_order[0]+bond_order[1]+bond_order[3]+bond_order[4])
            #2. Bsigpi
            descriptor[atom_id].append(bond_order[2]+bond_order[5]+bond_order[6]+bond_order[7])
        #for atom type 0~3, calculate Bsigsig and Bsigpi for all C-H bonds  (8)
        #for atom type 5, calculate Bsigsig, Bsigpi, and Bpipi for all connected bonds  (12)
        #overall, use all neighbors
        elif atom_symbol[atom_id-1] in ("C", "N"):
            bsigsig = []
            bsigpi = []
            bpipi = []
            for nbor_id in nbors_id:
                i=atom_id-1; j=nbor_id-1
                if nbor_id > atom_id:
                    i,j = j,i
                bond_order = bond_matrix[i][j]
                bsigsig.append(bond_order[0]+bond_order[1]+bond_order[3]+bond_order[4])
                bsigpi.append(bond_order[2]+bond_order[5]+bond_order[6]+bond_order[7])
                bpipi.append(bond_order[8])
            #1~4. Bsigsig_max, Bsigsig_min, Bsigsig_mean, Bsigsig_sum
            descriptor[atom_id].append(max(bsigsig))
            descriptor[atom_id].append(min(bsigsig))
            descriptor[atom_id].append(sum(bsigsig)/len(bsigsig))
            descriptor[atom_id].append(sum(bsigsig))
            #5~8. Bsigpi_max, Bsigpi_min, Bsigpi_mean, Bsigpi_sum
            descriptor[atom_id].append(max(bsigpi))
            descriptor[atom_id].append(min(bsigpi))
            descriptor[atom_id].append(sum(bsigpi)/len(bsigpi))
            descriptor[atom_id].append(sum(bsigpi))
            #9~12. Bpipi_max, Bpipi_min, Bpipi_mean, Bpipi_sum
            if atom_symbol[atom_id-1] == "N":
                descriptor[atom_id].append(max(bpipi))
                descriptor[atom_id].append(min(bpipi))
                descriptor[atom_id].append(sum(bpipi)/len(bpipi))
                descriptor[atom_id].append(sum(bpipi))
        #for atom type 6, calculate Bsigsig_OH,Bsigsig_OX,Bsigpi_OH,Bsigpi_OX,Bpipi_OX (5)
        else:
            oh = []  #Bsigsig, Bsigpi
            ox = []  #Bsigsig, Bsigpi, Bpipi
            for nbor_id in nbors_id:
                i=atom_id-1; j=nbor_id-1
                if nbor_id > atom_id:
                    i,j = j,i
                bond_order = bond_matrix[i][j]
                if atom_symbol[nbor_id-1] == "H":
                    oh.append(bond_order[0]+bond_order[1]+bond_order[3]+bond_order[4])
                    oh.append(bond_order[2]+bond_order[5]+bond_order[6]+bond_order[7])
                else:
                    ox.append(bond_order[0]+bond_order[1]+bond_order[3]+bond_order[4])
                    ox.append(bond_order[2]+bond_order[5]+bond_order[6]+bond_order[7])
                    ox.append(bond_order[8])
            assert len(oh)==2 and len(ox)==3
            #1~2. Bsigsig_OH, Bsigsig_OX
            descriptor[atom_id].append(oh[0])
            descriptor[atom_id].append(ox[0])
            #3~4. Bsigpi_OH, Bsigpi_OX
            descriptor[atom_id].append(oh[1])
            descriptor[atom_id].append(ox[1])
            #5. Bpipi_OX
            descriptor[atom_id].append(ox[2])
    # end of `for atom_id in bond_order.iterkeys():`

#called by 'calcTwoCenter'
def bondOrder(inf,descriptor,atom_nbors,atom_symbol):
    bond_order = [[] for _ in atom_symbol]  #lower triangle matrix
    line = inf.readline()
    while line!="" and ("DEGREES OF BONDING" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find \"DEGREES OF BONDING\""
        inf.close()
        sys.exit(1)
    line = inf.readline()  #empty line
    line = inf.readline()  #"0\n"
    while line.strip() != "":
        line = inf.readline()  #suppose to be "  C  1   C  2  C  3" or "0"
        while not line.startswith(" "):
            line = inf.readline()
        line = inf.readline()  #------
        line = inf.readline()
        while line.startswith(" "):
            line = line.split()
            atom_id = int(line[1])
            bond_order[atom_id-1].extend(map(float,line[2:]))
            line = inf.readline()
    for atom_id in descriptor.iterkeys():
        #get neighbors
        nbors = atom_nbors[atom_id]
        #get bond order between atom_id and nbors[i]
        bond = []
        for nbor in nbors:
            if atom_id > nbor:
                bond.append(bond_order[atom_id-1][nbor-1])
            else:
                bond.append(bond_order[nbor-1][atom_id-1])
        #for atom type 4: bond order of c-H
        if descriptor[atom_id][0] == 4:
            descriptor[atom_id].append(bond[0])
        #for atom type 0,1,2,3,5: bond order of all C-H all N-X, followed by max,min,mean,sum
        elif atom_symbol[atom_id-1] in ("C","N"):
            descriptor[atom_id].append(max(bond))
            descriptor[atom_id].append(min(bond))
            descriptor[atom_id].append(sum(bond)/len(bond))
            descriptor[atom_id].append(sum(bond))
        #for atom type 6: bond order of O-H and O-C
        else:
            boh = 0.; box = 0.
            if atom_symbol[nbors[0]-1] == "H":
                boh = bond[0]
            else:
                box = bond[0]
            if atom_symbol[nbors[1]-1] == "H":
                boh = bond[1]
            else:
                box = bond[1]
            descriptor[atom_id].append(boh)
            descriptor[atom_id].append(box)
    #end of bondOrder

#called by 'calcTwoCenter'
def twoCenterEnergy(inf,descriptor,atom_nbors,atom_symbol):
    line = inf.readline()
    while line!="" and ("TWO-CENTER TERMS" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find \"TWO-CENTER TERMS\""
        inf.close()
        sys.exit(1)
    line = inf.readline()
    while "ATOM" not in line:
        line = inf.readline()
    e2 = {}  #{atom_id:{nbor:energy,...},...}
    for atom_id in descriptor.iterkeys():
        e2[atom_id] = {}
        for nbor_id in atom_nbors[atom_id]:
            e2[atom_id][nbor_id] = None
    #to read total two-center energy of all related bonds
    no_block = 1
    first_id = 1
    second_id = 0
    while line.strip() != "":
        if line[3:7] == "ATOM":
            line = inf.readline()  #  PAIR 
            if no_block == 1:
                line = inf.readline()  # empty line: only appear in the first block?????
            line = inf.readline()
        first_id += 1
        second_id = 0
        while line.strip() != "":
            temp = line.split()
            second_id += 1
            if e2.has_key(first_id) and e2[first_id].has_key(second_id):
                e2[first_id][second_id] = float(temp[-1])
            if e2.has_key(second_id) and e2[second_id].has_key(first_id):
                e2[second_id][first_id] = float(temp[-1])
            line = inf.readline()
        no_block += 1
        line = inf.readline()
    #to calculate descriptors
    for atom_id in e2.iterkeys():
        nbors = e2[atom_id].keys()
        energy = e2[atom_id].values()
        #for atom type 4: calculate E2_cH
        if descriptor[atom_id][0] == 4:
            descriptor[atom_id].append(energy[0])
        #for atom type 0,1,2,3,5: calculate E2 for all C-H or N-X, followed by max,min,mean,sum
        elif atom_symbol[atom_id-1] in ("C","N"):
            descriptor[atom_id].append(max(energy))
            descriptor[atom_id].append(min(energy))
            descriptor[atom_id].append(sum(energy)/len(energy))
            descriptor[atom_id].append(sum(energy))
        #for atom type 6: calculate E2 for O-H and O-C
        else:
            eoh = 0.
            eox = 0.
            if atom_symbol[nbors[0]-1] == "H":
                eoh = energy[0]
            else:
                eox = energy[0]
            if atom_symbol[nbors[1]-1] == "H":
                eoh = energy[1]
            else:
                eox = energy[1]
            descriptor[atom_id].append(eoh)
            descriptor[atom_id].append(eox)
    #end of twoCenterEnergy

#called by 'calcDescriptor'
def calcTwoCenter(inf,descriptor,atom_nbors):
    '''
    to calculate those two-center terms.
    (bond order, total two-center energy)
    different atom type will yield different number of descriptors

    parameters
    ==========
    infile: string, .mopout file
    descriptor: dict. {atomid:[atom_type,descriptors...],...}
      after this function being called, `descriptor` will be updated.

    '''
    line = inf.readline()
    while line!= "" and ("CARTESIAN COORDINATES" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find \"CARTESIAN COORDINATES\""
        inf.close()
        sys.exit()
    line = inf.readline()  #empty line
    line = inf.readline()  #  NO.  ATOM X Y Z
    line = inf.readline()  #empty line
    line = inf.readline()
    atom_symbol = []
    while line.strip() != "":
        line = line.split()
        atom_symbol.append(line[1])
        line = inf.readline()
    
    #go to calculate SIGMA-PI BOND ORDER
    #2013-09-24 21:31   do not compute Bsigsig, Bsigpi, and Bpipi!!!!!!!!!!!!!!!!!
    sigmaPiBond(inf,descriptor,atom_nbors,atom_symbol)
    #go to calculate bond degree
    bondOrder(inf,descriptor,atom_nbors,atom_symbol)
    #total two-center energy
    twoCenterEnergy(inf,descriptor,atom_nbors,atom_symbol)
    return atom_symbol

#called by 'calcDescriptor'
def calcPolarAB(descriptor,eigenvector,mo_energy,homo_index,atom_nbors,atom_symbol):
    for atom_id in descriptor.iterkeys():
        polarAB = []
        for nbor in atom_nbors[atom_id]:
            polar = 0.
            for j in xrange(homo_index+1):
                for a in xrange(homo_index+1,len(mo_energy)):
                    value = 0.
                    for p in xrange(len(eigenvector[j][atom_id])):
                        for r in xrange(len(eigenvector[j][nbor])):
                            value += eigenvector[j][atom_id][p]*eigenvector[a][atom_id][p]*\
                                    eigenvector[j][nbor][r]*eigenvector[a][nbor][r]
                    value /= (mo_energy[j]-mo_energy[a])
                    polar += value
            polar *= 4.
            polarAB.append(polar)
        #for atom type 4: calculate polarAB_cH
        if descriptor[atom_id][0] == 4:
            descriptor[atom_id].append(polarAB[0])
        #for atom type 0,1,3,5: calculate polarAB for C-H or N-X, followed by max,min,mean,sum
        elif atom_symbol[atom_id-1] in ("C","N"):
            descriptor[atom_id].append(max(polarAB))
            descriptor[atom_id].append(min(polarAB))
            descriptor[atom_id].append(sum(polarAB)/len(polarAB))
            descriptor[atom_id].append(sum(polarAB))
        #for atom type 6: calculate polarAB for O-H and O-C
        else:
            poh = 0.
            pox = 0.
            if atom_symbol[atom_nbors[atom_id][0]-1] == "H":
                poh = polarAB[0]
            else:
                pox = polarAB[0]
            if atom_symbol[atom_nbors[atom_id][1]-1] == "H":
                poh = polarAB[1]
            else:
                pox = polarAB[1]
            descriptor[atom_id].append(poh)
            descriptor[atom_id].append(pox)
    #end of calcPolarAB

#called by calcSPAN
def floyd(mol):
    '''
    to calculate the length of shortest path between each atom
    using Floyd-Warshall algorithm.

    mol: <class 'pybel.Molecule'>
    '''
    inf = float("inf")
    #initialize
    dis = [[inf for i in xrange(len(mol.atoms))] for j in xrange(len(mol.atoms))]
    searched = [False for i in xrange(len(mol.atoms))]
    for i in xrange(len(searched)):
        dis[i][i] = 0
        searched[i] = True
        for nbor in pybel.ob.OBAtomAtomIter(mol.atoms[i].OBAtom):
            j = nbor.GetIdx()-1
            if searched[j] == True:
                continue
            dis[i][j] = 1
            dis[j][i] = 1
    #Floyd-Warshall algorithm
    n = len(mol.atoms)
    for k in xrange(n):
        for i in xrange(n):
            for j in xrange(n):
                if dis[i][k]==inf or dis[k][j]==inf:
                    continue
                if dis[i][k]+dis[k][j] < dis[i][j]:
                    dis[i][j] = dis[i][k]+dis[k][j]
    
    return dis

#called by 'calcDescriptor'
def calcSPAN(descriptor, mol):
    distance = floyd(mol)
    max_distance = 0.
    for atom_id in descriptor.iterkeys():
        dist = float(max(distance[atom_id-1]))
        if dist > max_distance:
            max_distance = dist
        descriptor[atom_id].append(dist)
    for atom_id in descriptor.iterkeys():
        descriptor[atom_id][-1] /= max_distance

def calcDescriptor(infile, candidate_atoms):
    '''
    to calculate descriptors

    parameter
    =========
    infile: string, mopout file
    candidate_atoms: None or list of atom ids. only calculate descriptors for those atoms

    return
    ======
    result: dict, key=atom_id, value=[atom_type,descriptors,...]
    '''
    mol = pybel.readfile("mopout",infile).next()
    #N-dealkylation, O-dealkylation, bezylic/allylic C-hydroxylation, aliphatic C-hydroxylation, 
    #aromatic c-hydroxylation, N-oxydation/Conjugation, O-Conjugation
    #[[atom1,atom2,...],[atom1,atom2,...],...]
    matches = matchAtoms(mol, candidate_atoms)
    #initialize `descriptors`
    descriptors = {}  #key=atom_id, value=[atom_type,descriptors,...]
    for i in xrange(len(matches)):
        for j in xrange(len(matches[i])):
            descriptors[matches[i][j]] = [i]
    #in case no candidate atoms being found
    if descriptors == {}:
        return descriptors
    #Attention: to make sure that only needed neighbors are included!!!!!!!!!!
    #for C, gathering all H neighbors
    #for O and N, gathering all neighbors
    atom_nbors = {}  #{atom:[nbors...],...}
    for atom in descriptors.iterkeys():
        atom_nbors[atom] = []
        for nbor in pybel.ob.OBAtomAtomIter(mol.atoms[atom-1].OBAtom):
            if (descriptors[atom][0] in range(5)) and nbor.IsHydrogen():
                atom_nbors[atom].append(nbor.GetIdx())
            elif descriptors[atom][0] in (5,6):
                atom_nbors[atom].append(nbor.GetIdx())
            else:
                pass
    #to process .mopout file
    inf = open(infile,"r")
    line = inf.readline()
    while line!="" and ("NO. OF FILLED LEVELS" not in line):
        line = inf.readline()
    if line == "":
        print "Error: can't find `NO. OF FILLED LEVELS` in %s"%infile
        inf.close()
        sys.exit()
    homo_index = int(line[line.rfind(" ")+1:])
    #RA,NA,EA,SEA,SNA,atomcQ,selfQ,activeQ
    eigenvector,mo_energy = calcOneCenter(inf,descriptors,homo_index-1)
    #sigmaPiBondOrder, bond order, total two-center energy
    inf.seek(0)
    atom_symbol = calcTwoCenter(inf,descriptors,atom_nbors)
    #polar_AB
    inf.seek(0)
    calcPolarAB(descriptors,eigenvector,mo_energy,homo_index-1,atom_nbors,atom_symbol)
    inf.close()
    #SPAN
    global do_span
    if do_span:
        calcSPAN(descriptors, mol)
    #logp
    global do_logp
    if do_logp:
        val = mol.calcdesc(["logP"])
        for key in descriptors.iterkeys():
            descriptors[key].append(val)

    return descriptors


#descriptor_names = ["RA","NA","EA","SEA","SNA","atomicQ","selfQ","activeQ",\
#        "Bsigsig","Bsigpi","Bpipi","B","E2","polarAB"]
def output(outf,value):
    global descriptor_names
    for i in xrange(8):
        print >>outf, "    %s: %g"%(descriptor_names[i],value[i+1])
    if value[0] in (0,1,2,3,5):
        j = 9
        for i in xrange(8,len(descriptor_names)):
            print >>outf, "    %s_max: %g"%(descriptor_names[i],value[j])
            print >>outf, "    %s_min: %g"%(descriptor_names[i],value[j+1])
            print >>outf, "    %s_mean: %g"%(descriptor_names[i],value[j+2])
            print >>outf, "    %s_sum: %g"%(descriptor_names[i],value[j+3])
            j += 4
    elif value[0] == 4:
        j = 9
        for i in xrange(8,len(descriptor_names)):
            print >>outf, "    %s_cH: %g"%(descriptor_names[i],value[j])
            j += 1
        #for i in xrange(8,10):
        #    print >>outf, "    %s_cH: %g"%(descriptor_names[i],value[j])
        #    j += 1
        #for i in xrange(11,len(descriptor_names)):
        #    print >>outf, "    %s_cH: %g"%(descriptor_names[i],value[j])
        #    j += 1
    else:
        j = 9
        for i in xrange(8,len(descriptor_names)):
            print >>outf, "    %s_OH: %g"%(descriptor_names[i],value[j])
            print >>outf, "    %s_OX: %g"%(descriptor_names[i],value[j+1])
            j += 2
        #for i in xrange(8,10):
        #    print >>outf, "    %s_OH: %g"%(descriptor_names[i],value[j])
        #    print >>outf, "    %s_OX: %g"%(descriptor_names[i],value[j+1])
        #    j += 2
        #print >>outf, "    %s_OX: %g"%(descriptor_names[11],value[j])
        #j += 1
        #for i in xrange(11,len(descriptor_names)):
        #    print >>outf, "    %s_OH: %g"%(descriptor_names[i],value[j])
        #    print >>outf, "    %s_OX: %g"%(descriptor_names[i],value[j+1])
        #    j += 2

#to calculate descriptors only for those atoms in `candidate_atoms`
def run(mopout_file, outf, candidate_atoms):
    '''
    parameters
    ==========
    mopout_file: string.
    outf: <type 'file'>. opened.
    candidate_atoms: None or list of atom ids.
    '''
    global DEBUG
    print >>sys.stderr, "to process %s..."%mopout_file,
    if candidate_atoms == []:
        results = {}
    else:
        results = calcDescriptor(mopout_file, candidate_atoms)
    if results == {}:
        print >>sys.stderr, "\n  it has no candidate atoms to be parsed!!",
    if DEBUG:
        print >>outf, "name:",mopout_file
        for key,value in results.iteritems():
            print >>outf, "  atom:",key
            print >>outf, "  type:",value[0]
            output(outf,value)
    else:
        print >>outf, mopout_file
        #outf.write(mopout_file)
        keys = results.keys()
        keys.sort()
        for key in keys:
            outf.write("\t%d:%d:%.6g"%(key,results[key][0],results[key][1]))
            for item in results[key][2:]:
                outf.write(",%.6g"%item)
            outf.write("\n")
    print >>sys.stderr, "Done"

def exit_with_help(prog_name):
    print "\n  Usage: %s [options] mopout out.txt"%prog_name
    print "  [options]"
    print "  -d      : if given, detailed description about descriptors will be shown"
    print "            instead of doing calculation"
    print "  -c file : file including candidate atoms for each molecule to be parsed."
    print "            each line of the file should be: `mol_name: atom1 atom2 ...`"
    print "  --span  : if given, calculate SPAN"
    print "  --logp  : if given, calculate logp and add to each atom"
    print "  mopout  : could be a .mopout file OR"
    print "            a file containing .mopout file in each line"
    print "  out.txt : where calculated descriptors will be saved.\n"
    sys.exit(1)

def load_candidate_file(infile):
    candidate = {}  #key=mol_name, value=[atom1,atom2,...]
    inf = open(infile,'r')
    for line in inf:
        mol_name,atoms = line.split(":")
        atoms = atoms.split()
        candidate[mol_name] = map(int,atoms)
    inf.close()
    return candidate

def main(argv=sys.argv):
    if len(argv) < 2:
        exit_with_help(argv[0])

    try:
        opt_list,args_list = getopt(argv[1:],"dc:",["span","logp"])
    except GetoptError,err:
        print err
        sys.exit(1)

    detail = False
    candidate_file = None
    mopout = ""
    outfile = ""
    #parse options
    global do_span
    global do_logp
    for opt,value in opt_list:
        if opt == "-d":
            detail = True
        elif opt == '-c':
            candidate_file = value
        elif opt == '--span':
            do_span = True
        elif opt == '--logp':
            do_logp = True
        else:
            print "Error: option %s not recognized"%opt
            sys.exit(1)

    #if to display detailed description about descriptors
    if detail:
        print description
        sys.exit(0)

    #get `mopout` and `outfile`
    if len(args_list) != 2:
        print "Error: invalid number of args"
        sys.exit(1)
    mopout = args_list[0]
    outfile = args_list[1]

    #load file containing candidate atoms for each molecule
    candidate = {}  #key=mol_name, value=[atom1,atom2,...]
    if candidate_file is not None:
        candidate = load_candidate_file(candidate_file)

    if os.path.isfile(mopout):
        outf = open(outfile,"w")
        if mopout.endswith(".mopout"):
            mol_name = os.path.basename(mopout).split(".")[0]
            try:
                candidate_atoms = candidate[mol_name]
            except KeyError:
                candidate_atoms = None
            run(mopout,outf,candidate_atoms)
        else:
            inf = open(mopout,"r")
            for line in inf:
                if line.startswith("#"):
                    continue
                if not line.strip().endswith(".mopout"):
                    print "Warning: invalid input file %s"%line.strip()
                    continue
                mol_name = os.path.basename(line).split(".")[0]
                try:
                    candidate_atoms = candidate[mol_name]
                except KeyError:
                    candidate_atoms = None
                run(line.strip(),outf,candidate_atoms)
            inf.close()
        outf.close()
    else:
        print "Error: %s is not a file!"%mopout


if __name__ == "__main__":
    main()

