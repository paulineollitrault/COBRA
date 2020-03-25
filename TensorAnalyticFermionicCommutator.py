import itertools as it
import logging
import sys
import copy
from qiskit.chemistry import FermionicOperator

import numpy as np

logger = logging.getLogger(__name__)

def quicksortL(arr):
    #quick sort just for arrays
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot ]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if pivot < x]
    return quicksortL(left) + middle + quicksortL(right)

def order(b):
    count = 0
    ideal = np.array(quicksortL(b))
    temp = b
    
    DoOrder = True
    
    while DoOrder==True:
        if np.array_equal(ideal,np.array(temp))==True:
            DoOrder = False

        for i in range(len(temp)-1):
            if temp[i]>temp[i+1]:
                tmp = temp[i]
                temp[i] = temp[i+1]
                temp[i+1] = tmp
                count +=1
                break
    return count

class term:
    coef = None
    oper = [] 
    stat = None
    
    def __init__(self,c=0,o=[],s=None):
        self.coef = c
        self.oper = o
        if(s=='fermi'): self.stat = -1
        else:           self.stat =  1

    def Print(self,message=''):
        s=message+' '+str(self.coef)+' * '
        for op in self.oper:
            if len(op)==2:
                s+=op[0]+'('+op[1]+')'
            elif len(op)==3:
                s+=op[0]+'('+op[1]+op[2]+')'
            else:
                s+='ErrorOp'
        print(s)

    def Copy(self):
        Y = term()
        Y.coef    = self.coef
        Y.oper    = [ok for ok in self.oper]
        Y.stat    = self.stat
        return Y        
        
    def Ordered(self):
        #Check if the term is Totally ordered
        expr  = [x[0] for x in self.oper]
        coefC  = [x[1] for x in self.oper if x[0]=='C']
        coefD  = [x[1] for x in self.oper if x[0]=='D']
        coefI  = [x[1] for x in self.oper if x[0]=='I']
        idenOD = [x[1]<=x[2] for x in self.oper if x[0]=='I']
        if (quicksortL(expr)==expr)and(quicksortL(coefC)==coefC)and(quicksortL(coefD)==coefD)and(quicksortL(coefI)==coefI)and(np.array(idenOD).all()==True):
            return True
        else:
            return False
    
    def CDIndexList(self):
        #gives a list of indicies for adag and a operators
        ls = ''
        for x in self.oper:
            if (x[0]=='C')or(x[0]=='D'):
                ls +=x[1]
        return ls
    def Normal_Ordered(self):
        expr = [x[0] for x in self.oper if (x[0]=='C')or(x[0]=='D')]
        if (quicksortL(expr)==expr):
            return True
        else:
            return False
    
    def IntOrder(self):
        Y = self.Copy()
        if (self.Normal_Ordered()==True)and(self.Ordered()==False):

            listC = [x[1] for x in self.oper if x[0]=='C']
            listD = [x[1] for x in self.oper if x[0]=='D']
            listI = [quicksortL([x[1],x[2]]) for x in self.oper if x[0]=='I']
            powerC = order(listC)
            listC  = quicksortL(listC)

            powerD = order(listD)
            listD  = quicksortL(listD)
            
            listI1 = [x[0] for x in listI]
            listI2 = quicksortL(listI1)
            listIden = []
            for x in listI2:
                listIden.append('I'+x+listI[listI1.index(x)][1])
    
            listCD = []
            for x in listC:
                listCD.append('C'+x)
            for x in listD:
                listCD.append('D'+x)
            Y.coef *=(Y.stat)**(powerC+powerD)
            Y.oper = listCD+listIden
            return Y.IntOrder()
        else:
            return Y
    def Power(self):
        coefC  = [x[1] for x in self.oper if x[0]=='C']        
        return len(coefC)
    def IdenTail(self):
        Y = self.Copy()
        listCD = [x for x in self.oper if (x[0]=='C')or(x[0]=='D')]
        listId = [x for x in self.oper if x[0]=='I']
        Y.oper = listCD+listId
        return Y
    def OpertoStr(self):
        res= ''
        for x in self.oper:
            res +=x
        return res
    
    def Product(self,O2,s):
        Y = term()
        Y.coef = self.coef*O2.coef*s
        Y.oper = self.oper+O2.oper
        Y.stat = self.stat
        return Y
    
class term_list:

    nop = 0
    eta = []

    def __init__(self,lst=[]):
        self.eta = lst
        self.nop = len(lst)
        
    def Remove_Zeros(self):
        self.nop = len(self.eta)
        idx      = [ i for i in range(self.nop) if self.eta[i].coef==0 ]
        for i in idx[::-1]:
            del self.eta[i]
        self.nop = len(self.eta)
    def Print(self):
        print("operator number ",self.nop)
        for x in self.eta:
            x.Print()
    def CDSimplify(self):
        self.Remove_Zeros()
        Done = (np.array([f.Normal_Ordered() for f in self.eta]).all()==True)
        if Done==True:
            return self
        elif Done==False:
            for i in range(self.nop):
                if self.eta[i].Normal_Ordered()==False:
                    x = self.eta[i].IdenTail()
                    for ind in range(len(x.oper)-1):
                        if (x.oper[ind][0]=='D')and(x.oper[ind+1][0]=='C'):
                            y  = x.Copy()
                            o1 = x.oper[ind]
                            o2 = x.oper[ind+1]
                            x.oper[ind]   = o2
                            x.oper[ind+1] = o1
                            x.coef *= x.stat
                        
                            y.oper.pop(ind+1)
                            y.oper.pop(ind)
                            y.oper.append('I'+o1[1]+o2[1])
                            break
                    self.eta[i] = x
                    self.eta.append(y)
                    self.nop = len(self.eta)
                    break
            return self.CDSimplify()
        
    def Simplify(self):
        #adag-a ordering
        self.CDSimplify()
        #internal ordering
        for i in range(self.nop):
            self.eta[i] = self.eta[i].IntOrder()
        #coef simplification
        OperLstSet = set([x.OpertoStr() for x in self.eta])
        if len(OperLstSet)<self.nop:
            neweta = []
            for elem in OperLstSet:
                newterm = term()
                for x in self.eta:
                    if x.OpertoStr()==elem:
                        newterm.coef +=x.coef
                        newterm.stat = x.stat
                        newterm.oper = x.oper
                neweta.append(newterm)
            self.eta = neweta
            self.nop = len(self.eta)
        self.Remove_Zeros()
        return self
    
    def Add(self,O2):
        C = term_list()
        C.nop = self.nop+O2.nop
        C.eta = self.eta + O2.eta
        C.Simplify()
        return C
 
    def Commutator(self,O2):
        C = term_list()
        nop1,nop2 = self.nop,O2.nop
        for m1 in range(nop1):
            for m2 in range(nop2):
                pxy = self.eta[m1].Product(O2.eta[m2], 1)
                myx = O2.eta[m2].Product(self.eta[m1],-1)
                C = C.Add(term_list([pxy]))
                C = C.Add(term_list([myx]))
                C.Simplify()
        return C
    
class rules:
    coef = None
    rule = ''
    power = None
    
    def __init__(self,c=0,r='',p=0):
        self.coef  = c
        self.rule  = r
        self.power = p
        
def make_rule(a,b,c):
    #function create a rule-class object with rule np.einsum
    #a,b->c; all a,b,c should be terms    
    ina = a.CDIndexList()
    inb = b.CDIndexList()
    out = c.CDIndexList()
    listIden = [x for x in c.oper if x[0]=='I']
    for x in listIden:
        ina = ina.replace(x[2],x[1])
        inb = inb.replace(x[2],x[1])        
    #rl = ina+','+inb+'->'+out
    #coef = c.coef
    #powr = c.Power()
    return rules(c.coef,ina+','+inb+'->'+out,c.Power())

def ComRules(o1,o2):
    #input: 2 terms; output : list of class-rule objects for np.einsum
    res= []
    l1   = term_list([o1])
    l2   = term_list([o2])
    lout = l1.Commutator(l2)
    for ind in range(lout.nop):
        res.append(make_rule(o1,o2,lout.eta[ind]))
    return res

def toPhys(h):
    #convert h-matrix from chemistry notations to physics one
    #now working only for 1 & 2 body terms
    if len(h.shape)==2:
        return h
    elif len(h.shape)==4:
        return np.einsum('ijkm->ikmj', h)
    else:
        return h

def toChem(h):
    #convert h-matrix from physics notations to chemistry one
    # now working only for 1 & 2 body terms
    if len(h.shape)==2:
        return h
    elif len(h.shape)==4:
        return np.einsum('ikmj->ijkm', h)
    else:
        return h

def hComPhys(ha,hb, stat = 'fermi', threshold=1e-12):
    #function which calculates commutator of two h-matricies
    res = []
    #latin alphabet for set of indicies
    alfa = 'abcdefghijklmnopqrstuvwxyz'
    # need to create two class-term  objects for each h-matrix
    dim_a = len(ha.shape)
    dim_b = len(hb.shape)
    
    oper_a = []
    oper_b = []
    
    for x in range(dim_a//2):
        oper_a.append('C'+alfa[0])
        alfa = alfa[1:]
    for x in range(dim_a//2):
        oper_a.append('D'+alfa[0])
        alfa = alfa[1:]
    for x in range(dim_b//2):
        oper_b.append('C'+alfa[0])
        alfa = alfa[1:]
    for x in range(dim_b//2):
        oper_b.append('D'+alfa[0])
        alfa = alfa[1:]
        
    term_a = term(1,oper_a,stat)
    term_b = term(1,oper_b,stat)
    
    CR = ComRules(term_a,term_b)
    
    PowList = set([x.power for x in CR])
    
    #-------------------#    
    for PW in PowList:
        #create empty tensor
        nferm = ha.shape[0]
        dim   = 2*PW
        cur   = np.zeros(np.full(dim,nferm))
        for x in CR:
            if x.power==PW:
                cur +=x.coef*np.einsum(x.rule,ha,hb,optimize=True)
        if np.abs(cur).max()>threshold:
            res.append(cur)
    return res

def hSimplify(h, stat = 'fermi', threshold=1e-12):

    print("I AM IN hSimplify")
    print("H in ",h.shape)
    res = np.zeros_like(h)
    if stat == 'fermi':
        eta = -1
    else:
        eta = +1
    

    print("RES ",res.shape)

    dim = len(h.shape)//2
    nferm = h.shape[0]

    print("I AM TRYING TO LOOP OVER ")
    print(len(list(it.combinations(np.arange(nferm), dim))))
    print("TWICE")

    # 1 -- ACCELERATE hSimplify REMOVING FOR LOOPS!
    
    for xL in it.combinations(np.arange(nferm), dim):
        for xR in it.combinations(np.arange(nferm), dim):
            val = 0
            for yL in it.permutations(xL):
                for yR in it.permutations(xR):
                    coef = eta**(order(np.array(yR))+order(np.array(yL)))
                    ind = list(yL)+list(yR)
                    val += coef*h[tuple(ind)]
            if np.abs(val)>threshold:
                res[tuple(list(xL)+list(xR))] = val
    return res

def mat_list_simplify(lst,nf=0,threshold=1e-12):
    res = [y for x in lst for y in x]
    if len(res)==0:
        return []
    else:
        PowList = set([len(x.shape) for x in res])
        res2 = []
        for PW in PowList:
            mt = np.zeros(np.full(PW,nf))
            for elem in res:
                if len(elem.shape)==PW:
                    mt +=elem
            if np.abs(mt).max()>threshold:
                res2.append(mt)
        return res2
    
def ten_commutator(fop_a, fop_b, fop_c=None, stat = 'fermi', Chem=True, threshold=1e-12):
    # fop_a, fop_b, fop_c - Fermionic Operators
    # if fop_c ==0 => return = [a,b] ([X,Y]=X*Y-Y*X)
    # if fop_c !=0 => return = 0.5*([[a,b],c]+[a,[b,c]])
    
    ha_list = []
    if np.all(fop_a.h1)!=0:
        ha_list.append(fop_a.h1)
    if np.all(fop_a.h2)!=0:
        ha_list.append(fop_a.h2)
    
    hb_list = []
    if np.all(fop_b.h1)!=0:
        hb_list.append(fop_b.h1)
    if np.all(fop_b.h2)!=0:
        hb_list.append(fop_b.h2)
    
    hc_list = []
    if fop_c is not None:
        if np.all(fop_c.h1)!=0:
            hc_list.append(fop_c.h1)
        if np.all(fop_c.h2)!=0:
            hc_list.append(fop_c.h2)

    ha_phys_list = []
    hb_phys_list = []
    hc_phys_list = []
    
    if Chem==True:
        for x in ha_list:
            ha_phys_list.append(toPhys(x))
        for x in hb_list:
            hb_phys_list.append(toPhys(x))
        for x in hc_list:
            hc_phys_list.append(toPhys(x))

    if len(ha_phys_list)!=0:
        nf = ha_phys_list[0].shape[0]
    else:
        nf = hb_phys_list[0].shape[0]
            
    if fop_c is None:
        #just [A,B]
        res = []
        for x in ha_phys_list:
            for y in hb_phys_list:
                res.append(hComPhys(x,y, stat, threshold))
        
        res = mat_list_simplify(res,nf,threshold)
        
        if len(res)==0:
            #return empty fermionic operator
            return FermionicOperator(np.zeros((nf,nf)))
        else:
            h1out = np.zeros((nf,nf))
            h2out = np.zeros((nf,nf,nf,nf))

            for x in res:
                if len(x.shape)==2:
                    h1out = hSimplify(x, stat, threshold)
                if len(x.shape)==4:
                    h2out = hSimplify(x, stat, threshold)
            if Chem==True:
                return FermionicOperator(toChem(h1out),toChem(h2out))
            else:
                return FermionicOperator(h1out,h2out)                    
    else:
        #([[A,B],C]+[A,[B,C]])/2
        comAB = []
        for x in ha_phys_list:
            for y in hb_phys_list:
                comAB.append(hComPhys(x,y, stat, threshold))
        comAB = mat_list_simplify(comAB,nf,threshold)
        if len(comAB)==0:
            comAB_C = []
        else:
            comAB_C = []
            for x in comAB:
                for y in hc_phys_list:
                    comAB_C.append(hComPhys(x,y, stat, threshold))
        
        comBC = []
        for x in hb_phys_list:
            for y in hc_phys_list:
                comBC.append(hComPhys(x,y, stat, threshold))
        comBC = mat_list_simplify(comBC,nf,threshold)
        if len(comBC)==0:
            comA_BC = []
        else:
            comA_BC = []
            for x in ha_phys_list:
                for y in comBC:
                    comA_BC.append(hComPhys(x,y, stat, threshold))
        
        comABC = mat_list_simplify(comAB_C+comA_BC,nf,threshold)

        if len(comABC)==0:
            #return empty fermionic operator
            return FermionicOperator(np.zeros((nf,nf)))
        else:
            h1out = np.zeros((nf,nf))
            h2out = np.zeros((nf,nf,nf,nf))

            for x in comABC:
                if len(x.shape)==2:
                    h1out = 0.5*hSimplify(x, stat, threshold)
                if len(x.shape)==4:
                    h2out = 0.5*hSimplify(x, stat, threshold)
            if Chem==True:
                return FermionicOperator(toChem(h1out),toChem(h2out))
            else:
                return FermionicOperator(h1out,h2out)
