import numpy as np
import itertools

from operator_string import *

def overwrite(s,i,x):
    l=list(s)
    l[i]=x
    return ''.join(l)

def decorate(s):
    if(s[0]=='C'): return 'C('+s[1]+')'
    else:          return 'D('+s[1]+')'

def findsubset(s,ln):
    return list(itertools.combinations(s,ln))

# ========== #

class Operator_String_List:

    nop = 0
    eta = []

    def __init__(self,lst=[]):
        self.eta = lst
        self.nop = len(lst)

    def Remove_Zeros(self):
        self.nop = len(self.eta)
        idx      = [ i for i in range(self.nop) if self.eta[i].coefficient==0 ]
        for i in idx[::-1]:
            del self.eta[i]
        self.nop = len(self.eta)

    def Simplify(self):

        self.Remove_Zeros()
        for i in range(self.nop):
            self.eta[i].Tensor_Simplification()

        i=0
        while(i<len(self.eta)):
         matches=[]
         for j in range(i+1,len(self.eta)):
             ti = self.eta[i].tensors
             tj = self.eta[j].tensors
             oi = self.eta[i].operators
             oj = self.eta[j].operators
             if(ti==tj and oi==oj): matches.append(j)
         for j in matches:
             self.eta[i].coefficient += self.eta[j].coefficient
             self.eta[j].coefficient  = 0
         i+=1
        self.nop = len(self.eta)   

        self.Remove_Zeros()
        for i in range(self.nop):
            self.eta[i].Tensor_Simplification()
        self.nop = len(self.eta)
            
    def Print(self,option='terminal',output_file=None):

        for i in range(len(self.eta)):
            rk_i = len(self.eta[i].operators)//2-1
            for j in range(i+1,len(self.eta)):
                rk_j = len(self.eta[j].operators)//2-1
                if(rk_i>rk_j):
                   tmp_i = (self.eta[i]).Copy()
                   tmp_j = (self.eta[j]).Copy()
                   self.eta[i],self.eta[j] = tmp_j,tmp_i

        if(option=='python' and output_file is not None):
           operators_rank = [len(x.operators)//2-1 for x in self.eta]
           for rk in list(set(operators_rank)):
               output_file.write('       hs[%d]  = np.zeros(tuple([n]*%d))\n' % (rk,2*(rk+1)))
               output_file.write('       idx[%d] = []\n' % rk)
        for x in self.eta:
            x.Print(option,output_file)

    def Remove_OV_terms(self,occ_list,vrt_list):
        idx = []
        for i in occ_list:
            for v in vrt_list:
                Iiv,Ivi = 'I'+i+v,'I'+v+i
                for k in range(self.nop):
                    if(Iiv in self.eta[k].tensors or Ivi in self.eta[k].tensors):
                       if(k not in idx): idx.append(k)
        idx=sorted(idx)
        for i in idx[::-1]:
            del self.eta[i]
        self.nop = len(self.eta)

    def Commutator(self,O2):
        C = Operator_String_List(lst=[])
        nop1,nop2 = self.nop,O2.nop
        for m1 in range(nop1):
            for m2 in range(nop2):
                pxy = self.eta[m1].Product(O2.eta[m2], 1)
                myx = O2.eta[m2].Product(self.eta[m1],-1)
                C.eta += pxy.Wick_Decompose()
                C.eta += myx.Wick_Decompose()
        C.Simplify()
        return C

    def Add(self,O2):
        C = Operator_String_List(lst=[])
        C.nop = self.nop + O2.nop
        C.eta = self.eta + O2.eta
        C.Simplify()
        return C

