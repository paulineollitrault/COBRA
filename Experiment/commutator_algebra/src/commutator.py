import numpy as np
import itertools

def overwrite(s,i,x):
    l=list(s)
    l[i]=x
    return ''.join(l)

def decorate(s):
    if(s[0]=='C'): return 'C('+s[1]+')'
    else:          return 'D('+s[1]+')'

# ========== #

class Operator_String:
    coeff     = None
    tensors   = []
    operators = [] 
    stat      = None

    def __init__(self,c=0,t=[],o=[],s=None):
        self.coeff     = c
        self.tensors   = t
        self.operators = o
        if(s=='fermi'): self.stat = -1
        else:           self.stat =  1

    def Print(self,message=''):
        s=message+' '+str(self.coeff)+' * '
        for t in self.tensors:   s+=t+' '
        s+='* '
        for o in self.operators: s+=decorate(o)+' '
        print(s)

    def Copy(self):
        Y = Operator_String()
        Y.coeff     = self.coeff
        Y.tensors   = [tk for tk in self.tensors]
        Y.operators = [ok for ok in self.operators]
        return Y

    def Normal_Order(self):
        Y = self.Copy()
        nop = len(Y.operators)
        for i in range(nop):
            for j in range(i+1,nop):
                oi,oj = Y.operators[i],Y.operators[j]
                if(oi[0]=='D' and oj[0]=='C'):
                   Y.coeff *= self.stat
                   Y.operators[i],Y.operators[j] = oj,oi
                if(oi[0]==oj[0] and oj[1]<oi[1]):
                   Y.coeff *= self.stat
                   Y.operators[i],Y.operators[j] = oj,oi
        return Y

    def Contract(self,i,j):
        minij,maxij=min(i,j),max(i,j)
        oi,oj = self.operators[minij],self.operators[maxij]
        if(oi[0]=='D' and oj[0]=='C'):
           Y = self.Copy()
           Y.tensors   += ['I'+oi[1]+oj[1]]
           del Y.operators[maxij]
           del Y.operators[minij]
           return Y
        else:
           return Operator_String()

    def Tensor_Simplification(self):
        contract = True
        while(contract):
          nt  = len(self.tensors)
          idx = [k for k in range(nt) if self.tensors[k][0]=='I']
          # are there identities?
          if(len(idx)==0):
             contract = False
          else:
             i1,j1,j2 = idx[0],None,None
             for je in range(nt):
                 if(je!=i1 and self.tensors[i1][1] in self.tensors[je][1:]): j1=je
                 if(je!=i1 and self.tensors[i1][2] in self.tensors[je][1:]): j2=je
             # can these be contracted?
             if(j1 is None):
                if(j2 is None): 
                   contract = False
                else:
                   for mu,x in enumerate(self.tensors[j2][1:]):
                       if(x==self.tensors[i1][2]): self.tensors[j2]=overwrite(self.tensors[j2],mu+1,self.tensors[i1][1])
                   del self.tensors[i1]
             else:
                for mu,x in enumerate(self.tensors[j1][1:]):
                    if(x==self.tensors[i1][1]): self.tensors[j1]=overwrite(self.tensors[j1],mu+1,self.tensors[i1][2])
                del self.tensors[i1]

    def Wick_Decompose(self):
        nop = len(self.operators)
        wick_list = [self.Normal_Order()]
        for (i,j) in itertools.combinations(range(nop),2):
            wick_list += self.Contract(i,j).Normal_Order().Wick_Decompose()
        return wick_list

    def Product(self,O2,s):
        Y = Operator_String()
        Y.coeff     = self.coeff*O2.coeff*s
        Y.tensors   = sorted(self.tensors+O2.tensors)
        Y.operators = self.operators+O2.operators
        return Y

# ========== #

class Operator_String_List:

    nop = 0
    eta = []

    def __init__(self,lst=[]):
        self.eta = lst
        self.nop = len(lst)

    def Remove_Zeros(self):
        self.nop = len(self.eta)
        idx      = [ i for i in range(self.nop) if self.eta[i].coeff==0 ]
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
             self.eta[i].coeff += self.eta[j].coeff
             self.eta[j].coeff  = 0
         i+=1
        self.nop = len(self.eta)   

        self.Remove_Zeros()
        for i in range(self.nop):
            self.eta[i].Tensor_Simplification()
        self.nop = len(self.eta)
            
    def Print(self):
        print("operator number ",self.nop)
        for x in self.eta:
            x.Print()

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

    '''
    def Product(self,O2,s):
        Y = Operator_String()
        Y.coeff     = self.coeff*O2.coeff*s
        Y.tensors   = sorted(self.tensors+O2.tensors)
        Y.operators = self.operators+O2.operators
        return Y
    '''
 
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
        C.nop = self.nop+O2.nop
        C.eta = self.eta + O2.eta
        C.Simplify()
        return C

# ========== #

def fixme_asap(L,s):
    if(len(L)==1):   X=np.einsum(s,L[0],optimize=True)
    elif(len(L)==2): X=np.einsum(s,L[0],L[1],optimize=True)
    elif(len(L)==3): X=np.einsum(s,L[0],L[1],L[2],optimize=True)
    elif(len(L)==4): X=np.einsum(s,L[0],L[1],L[2],L[3],optimize=True)
    else:            assert(False)
    return X

def matrix_elements(C,matrices,nb):
    C_matrix = None
    for mu,ci in enumerate(C.eta):
        cof,ten,lst = ci.coeff,ci.tensors,ci.operators
        if(C_matrix is None): C_matrix=np.zeros([nb]*len(lst))
        ten_names = [t[0]  for t in ten]
        ten_index = [t[1:] for t in ten]
        einsum_string  = ''.join(s+',' for s in ten_index)
        einsum_string  = einsum_string[:-1]+'->'
        einsum_string += ''.join(s[1] for s in lst)
        C_matrix += cof*fixme_asap([matrices[x] for x in ten_names],einsum_string)

    return C_matrix
