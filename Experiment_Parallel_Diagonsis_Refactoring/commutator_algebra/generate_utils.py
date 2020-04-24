import sys
sys.path.append('./src/')
from operator_string      import *
from operator_string_list import *

def adjoint(op):
    adj = []
    for a in op[::-1]:
        if(a[0]=='C'): adj.append("".join(['D',a[1]]))
        if(a[0]=='D'): adj.append("".join(['C',a[1]]))
    return adj

def print_excitations(Emu,Enu,adj,outf):
  outf.write('    if(len(Emu)==%d and len(Enu)==%d):\n' % (len(Emu),len(Enu)))
  idx_mu = ",".join([s[1] for s in Emu])
  idx_nu = ",".join([s[1] for s in Enu])
  outf.write('       %s = Emu\n' % idx_mu)
  outf.write('       %s = Enu\n' % idx_nu)
  Emu = adjoint(Emu)
  if(adj): Enu = adjoint(Enu)
  X = Operator_String_List(lst=[Operator_String(1,[],Emu,'fermi')])
  Y = Operator_String_List(lst=[Operator_String(1,[],Enu,'fermi')])
  CXY = X.Commutator(Y)
  CXY.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
  CXY.Simplify()
  CXY.Print('python',outf)

def print_one_body(Emu,Enu,adj,outf):
  outf.write('    if(len(Emu)==%d and len(Enu)==%d):\n' % (len(Emu),len(Enu)))
  idx_mu = ",".join([s[1] for s in Emu])
  idx_nu = ",".join([s[1] for s in Enu])
  outf.write('       %s = Emu\n' % idx_mu)
  outf.write('       %s = Enu\n' % idx_nu)
  Emu = adjoint(Emu)
  if(adj): Enu = adjoint(Enu)
  X = Operator_String_List(lst=[Operator_String(1,[     ],        Emu,'fermi')])
  Y = Operator_String_List(lst=[Operator_String(1,['Hpq'],['Cp','Dq'],'fermi')])
  Z = Operator_String_List(lst=[Operator_String(1,[     ],        Enu,'fermi')])
  CXY = X.Commutator(Y)
  C_CXY_Z = CXY.Commutator(Z)
  C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
  CYZ = Y.Commutator(Z)
  C_X_CYZ = X.Commutator(CYZ)
  C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
  C = C_CXY_Z.Add(C_X_CYZ)
  C.Simplify()
  C.Print('python',outf)

def print_two_body(Emu,Enu,adj,outf):
  outf.write('    if(len(Emu)==%d and len(Enu)==%d):\n' % (len(Emu),len(Enu)))
  idx_mu = ",".join([s[1] for s in Emu])
  idx_nu = ",".join([s[1] for s in Enu])
  outf.write('       %s = Emu\n' % idx_mu)
  outf.write('       %s = Enu\n' % idx_nu)
  Emu = adjoint(Emu)
  if(adj): Enu = adjoint(Enu)
  X = Operator_String_List(lst=[Operator_String(1,[     ],                    Emu,'fermi')])
  Y = Operator_String_List(lst=[Operator_String(1,['Hprqs'],['Cp','Cq','Ds','Dr'],'fermi')])
  Z = Operator_String_List(lst=[Operator_String(1,[     ],                    Enu,'fermi')])
  CXY = X.Commutator(Y)
  C_CXY_Z = CXY.Commutator(Z)
  C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
  CYZ = Y.Commutator(Z)
  C_X_CYZ = X.Commutator(CYZ)
  C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
  C = C_CXY_Z.Add(C_X_CYZ)
  C.Simplify()
  C.Print('python',outf)

#=========================================================#

outf = open('utils.py','w')

outf.write('import numpy as np\n')
outf.write('import itertools\n')
outf.write('from qiskit.chemistry import FermionicOperator\n')
outf.write('from fermionic_operator_nbody import FermionicOperatorNBody\n')
outf.write('\n')
outf.write('def delta(p,r):\n')
outf.write('    if(p==r): return 1\n')
outf.write('    return 0\n')
outf.write('\n')
outf.write('def commutator_adj_nor(n,Emu,Enu):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_excitations(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_excitations(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_excitations(excitation_mu,excitation_nu,False,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

outf.write('\n')
outf.write('def commutator_adj_adj(n,Emu,Enu):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_excitations(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_excitations(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_excitations(excitation_mu,excitation_nu,True,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

outf.write('\n')
outf.write('def triple_commutator_adj_onebody_nor(n,Emu,Enu,H):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_one_body(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_one_body(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_one_body(excitation_mu,excitation_nu,False,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

outf.write('\n')
outf.write('def triple_commutator_adj_onebody_adj(n,Emu,Enu,H):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_one_body(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_one_body(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_one_body(excitation_mu,excitation_nu,True,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

outf.write('\n')
outf.write('def triple_commutator_adj_twobody_nor(n,Emu,Enu,H):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_two_body(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_two_body(excitation_mu,excitation_nu,False,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_two_body(excitation_mu,excitation_nu,False,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

outf.write('\n')
outf.write('def triple_commutator_adj_twobody_adj(n,Emu,Enu,H):\n')
outf.write('    hs  = [None]*4\n')
outf.write('    idx = [None]*4\n')

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Dj']
print_two_body(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Di'],['Cb','Cc','Dk','Dj']
print_two_body(excitation_mu,excitation_nu,True,outf)

excitation_mu,excitation_nu = ['Ca','Cb','Dj','Di'],['Cc','Cd','Dl','Dk']
print_two_body(excitation_mu,excitation_nu,True,outf)

outf.write('    for k in range(4):\n')
outf.write('        if(idx[k] is not None): idx[k] = list(set(idx[k]))\n')
outf.write('    return FermionicOperatorNBody(hs),idx\n')

#=========================================================#

