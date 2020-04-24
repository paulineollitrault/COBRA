import numpy as np
import string
import itertools

def overwrite(s,i,x):
    l=list(s)
    l[i]=x
    return ''.join(l)

def decorate(s):
    if(s[0]=='C'): return 'C('+s[1]+')'
    else:          return 'D('+s[1]+')'

def findsubset(s,ln):
    return list(itertools.combinations(s,ln))

dummies         = ['p','r','q','s']
dummy_reservoir = ['u','v','w','x','y','z']

class Operator_String:
      coefficient = 0
      tensors     = []
      operators   = []
      statistics  = 'fermi'

      def __init__(self,coefficient=0,tensors=[],operators=[],statistics='fermi'):
          self.coefficient = coefficient
          self.tensors     = tensors
          self.operators   = operators
          self.statistics  = statistics

      def n_tensors(self):
          return len(self.tensors)

      def n_operators(self):
          return len(self.operators)
      
      def rank(self):
          return len(self.operators)//2-1
 
      def statistics_to_sign(self):
          if(self.statistics=='fermi'): return -1
          else:                         return  1

      def transform_dummies(self,string,substitutor):
          output = string
          for dummy_index in dummies: output = output.replace(dummy_index,substitutor)
          return output

      def permute(self,target,mask):
          nop = len(target)
          idx = [x for x in range(nop)]
          swp = 0
          for i in range(nop):
              for j in range(i+1,nop):
                  idx_i = mask.index(target[i][0])
                  idx_j = mask.index(target[j][0])
                  if(idx_i>idx_j):
                     swp += 1
                     idx[i],idx[j] = idx[j],idx[i]
                     target[i],target[j] = target[j],target[i]
          if(swp==0): return ''
          else:       return '.transpose('+','.join([str(i) for i in idx])+')'

      def Print(self,option='terminal',output_file=None):

          if(option=='terminal'):
             print("Coefficient: ",self.coefficient)
             print("Tensors:     ",self.tensors)
             print("Operators:   ",self.operators)
             print("Statistics:  ",self.statistics)
          elif(option=='python'):
             label = '       hs['+str(self.rank())+']['
             # switch to chemist's notation
             operator_indices_chemist = [None]*self.n_operators()
             m = 0
             for i in range(self.n_operators()//2):
                 operator_indices_chemist[m] = self.operators[i]; m += 2
             m =1
             for i in range(self.n_operators()//2,self.n_operators())[::-1]:
                 operator_indices_chemist[m] = self.operators[i]; m += 2

             list_of_operator_dummies = [o[1] for o in operator_indices_chemist if o[1] in dummies]

             list_of_tensor_dummies = []
             for k,t in enumerate(self.tensors):
                   for i,s in enumerate(list(t[1:])):
                       if(s in dummies): list_of_tensor_dummies.append((s,k,i))

             permutation_string = ''
             if(len(list_of_tensor_dummies)>1):
                permutation_string = self.permute(list_of_tensor_dummies,list_of_operator_dummies)

             list_of_operator_indices = [ self.transform_dummies(o[1],':') for o in operator_indices_chemist]
             label = label + ','.join(list_of_operator_indices) + '] += '+str(self.coefficient)

             # add tensors
             for k,t in enumerate(self.tensors):
                 if(t[0]=='I'): t_name = 'delta('; t_tail = ')'
                 else:          t_name = t[0]+'['; t_tail = ']'
                 if(k in [a[1] for a in list_of_tensor_dummies]): label += '*'+t_name+','.join([self.transform_dummies(x,':') for x in t[1:]])+t_tail+permutation_string
                 else:                                            label += '*'+t_name+','.join([self.transform_dummies(x,':') for x in t[1:]])+t_tail

             nu,used_dummy = 0,[]
             for mu,k in enumerate(list_of_operator_indices):
                 if(k==':'):
                    list_of_operator_indices[mu] = dummy_reservoir[nu]; used_dummy.append(dummy_reservoir[nu]); nu += 1

             if(nu==0):   label += '; idx['+str(self.rank())+'] += [('+','.join(list_of_operator_indices)+')]'
             elif(nu==1): label += '; idx['+str(self.rank())+'] += [('+','.join(list_of_operator_indices)+') for '+used_dummy[0]+' in range(n)]'
             else:        label += '; idx['+str(self.rank())+'] += [('+','.join(list_of_operator_indices)+') for ('+','.join(used_dummy)+') in itertools.product(range(n),repeat='+str(nu)+')]'

             if(output_file is None): print(label)
             else:                    output_file.write(label+'\n')


      def Copy(self):
          Y = Operator_String()
          Y.coefficient = self.coefficient
          Y.tensors     = [tk[:] for tk in self.tensors]
          Y.operators   = [ok[:] for ok in self.operators]
          Y.statistics  = self.statistics
          return Y
      
      def Tensor_Simplification(self):

          continue_to_contract = True

          while(continue_to_contract):

                # find Kronecker/non-Kronecker tensors
                deltas = [k for k in range(self.n_tensors()) if self.tensors[k][0]=='I']
                others = [k for k in range(self.n_tensors()) if self.tensors[k][0]!='I']
            
                # find all indices that can be contracted
                deltas_indices = {}
      
                for k in deltas:
                    for i,s in enumerate(list(self.tensors[k][1:])):
                        if(s not in deltas_indices.keys()): deltas_indices[s] = [(k,i)]
                        else:                               deltas_indices[s].append((k,i))
      
                for k in others:
                    for i,s in enumerate(list(self.tensors[k][1:])):
                        if(s in deltas_indices.keys()):     deltas_indices[s].append((k,i))
     
                # stop if there are no deltas 
                if(len(deltas_indices.keys())==0):
                   continue_to_contract = False

                else:
                   contraction = [k for k in deltas_indices.keys() if len(deltas_indices[k])>1]
                   # stop if there are no repeated indices
                   if(len(contraction)==0):
                      continue_to_contract = False
                   else:
                      contraction = contraction[0]
                      tensor_i,position_i = deltas_indices[contraction][0]
                      tensor_j,position_j = deltas_indices[contraction][1]
                      # contraction between deltas
                      if(self.tensors[tensor_i][0]=='I' and self.tensors[tensor_j]=='I'):
                         self.tensors[tensor_j]='I'+self.tensors[tensor_j][1+(1+position_j)%2]+self.tensors[tensor_i][1+(1+position_i)%2]
                         del self.tensors[tensor_i]
                      # contraction between delta and non-delta
                      if(self.tensors[tensor_i][0]=='I' and self.tensors[tensor_j]!='I'):
                         self.tensors[tensor_j] = self.tensors[tensor_j][:1+position_j] + self.tensors[tensor_i][1+(1+position_i)%2] + self.tensors[tensor_j][position_j+2:]
                         del self.tensors[tensor_i]

      def Contract(self,ci,cj):
          if(len(self.operators)==0): return Operator_String()
          i,j = self.operators.index(ci),self.operators.index(cj)
          minij,maxij=min(i,j),max(i,j)
          oi,oj = self.operators[minij],self.operators[maxij]
          if(oi[0]=='D' and oj[0]=='C'):
             Y = self.Copy()
             Y.tensors     += ['I'+oi[1]+oj[1]]
             Y.coefficient *= self.statistics_to_sign()**(maxij-minij-1)
             del Y.operators[maxij]
             del Y.operators[minij]
             return Y
          else:
             return Operator_String()

      def Normal_Order(self):
          Y = self.Copy()
          for i in range(self.n_operators()):
              for j in range(i+1,self.n_operators()):
                  oi,oj = Y.operators[i],Y.operators[j]
                  if( (oi[0]=='D' and oj[0]=='C') or (oi[0]==oj[0] and oj[1]<oi[1])):
                     Y.coefficient *= self.statistics_to_sign()
                     Y.operators[i],Y.operators[j] = oj,oi
          return Y

      def Product(self,O2,s):
          Y = Operator_String()
          Y.coefficient = self.coefficient*O2.coefficient*s
          Y.tensors     = sorted(self.tensors+O2.tensors)
          Y.operators   = self.operators+O2.operators
          Y.statistics  = self.statistics
          return Y

      def Wick_Decompose(self):
          nop = len(self.operators)
          wick_list = []
          ncr = nop//2
          crt = [oi for oi in self.operators if oi[0]=='C']
          dst = [oi for oi in self.operators if oi[0]=='D']
          for k in range(ncr+1):
              for gc,fd in itertools.product(findsubset(crt,k),findsubset(dst,k)):
                  for sfd in itertools.permutations(fd):
                      op = self.Copy()
                      for ci,di in zip(gc,sfd):
                          op = op.Contract(ci,di)
                      op = op.Normal_Order()
                      wick_list.append(op)
          return wick_list

