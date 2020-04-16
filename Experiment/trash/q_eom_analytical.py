import numpy as np
from qiskit.chemistry import FermionicOperator
#from TensorAnalyticFermionicCommutator import ten_commutator
from fermionic_operator_nbody import FermionicOperatorNBody

def delta(p,r):
    if(p==r): return 1
    return 0


class q_EOM:

      def __init__(self,n,nelec,h1,h2):
          self.n     = n
          self.nelec = nelec
          self.htot  = 0.5*h2[:,:,:,:]+np.einsum('pr,qs->pqsr',h1,np.eye(n)/(self.nelec-1))

      def build_excitation_operators(self):
          no,nv     = self.nelec,self.n-self.nelec
          self.nexc = no*nv + no*(no-1)//2 * nv*(nv-1)//2
          self.E_mu = []
          for i in range(self.nelec):
              for a in range(self.nelec,self.n):
                  self.E_mu.append([a,i])
          for i in range(self.nelec):
              for j in range(i+1,self.nelec):
                  for a in range(self.nelec,self.n):
                      for b in range(a+1,self.n):
                          self.E_mu.append([a,b,i,j])

      def commutator_adj_nor(self,Emu,Enu):
          hs = [np.zeros(tuple([self.n]*s)) for s in [2,4,6,8]]
          if(len(Emu)==2 and len(Enu)==2):
             a,i = Emu
             b,j = Enu
             hs[0][i,j] =  delta(a,b)
             hs[0][b,a] = -delta(i,j)
          if(len(Emu)==2 and len(Enu)==4):
             a,i     = Emu
             b,c,k,j = Enu
             hs[1][c,i,j,k] =  delta(a,b)
             hs[1][b,i,j,k] =  delta(a,c)
             hs[1][b,c,a,j] = -delta(k,i)
             hs[1][b,c,a,k] = -delta(j,i)
          if(len(Emu)==4 and len(Enu)==2):
             a,b,j,i = Emu
             c,k     = Enu
             hs[1][i,j,a,k] =  delta(b,c)
             hs[1][i,j,b,k] =  delta(a,c)
             hs[1][c,j,a,b] = -delta(k,i)
             hs[1][c,i,a,b] = -delta(k,j)
          if(len(Emu)==4 and len(Enu)==4):
             a,b,j,i = Emu
             c,d,l,k = Enu
             hs[2][d,i,j,a,k,l] =  delta(b,c)
             hs[2][c,i,j,a,k,l] =  delta(b,d)
             hs[2][d,i,j,b,k,l] =  delta(a,c)
             hs[2][c,i,j,b,k,l] =  delta(a,d)
             hs[2][c,d,j,a,b,k] = -delta(l,i)
             hs[2][c,d,i,a,b,k] = -delta(l,j)
             hs[2][c,d,j,a,b,l] = -delta(k,i)
             hs[2][c,d,i,a,b,l] = -delta(k,j)
          return FermionicOperatorNBody(hs)

      def nested_commutator_adj_nor(self,Emu,Enu):
          hs = [np.zeros(tuple([self.n]*s)) for s in [2,4,6,8]]
          H  = self.htot[:,:,:,:]
          if(len(Emu)==2 and len(Enu)==2):
             a,i = Emu
             b,j = Enu
             hs[1][i,:,j,:] =  2 * H[a,:,b,:] 
             hs[1][i,:,j,:] =  2 * H[a,:,:,b] 
             hs[1][b,:,:,:] = -1 * H[a,:,:,:] * delta(j,i) 
             hs[1][b,i,:,:] = -2 * H[a,j,:,:] 
             hs[1][i,:,j,:] =  2 * H[:,a,b,:] 
             hs[1][i,:,j,:] =  2 * H[:,a,:,b] 
             hs[1][b,:,:,:] = -1 * H[:,a,:,:] * delta(j,i) 
             hs[1][b,i,:,:] = -2 * H[j,a,:,:] 
             hs[1][:,:,j,:] = -1 * H[:,:,:,i] * delta(a,b) 
             hs[1][:,:,a,j] = -2 * H[:,:,b,i] 
             hs[1][b,:,a,:] =  2 * H[j,:,:,i] 
             hs[1][b,:,a,:] =  2 * H[:,j,:,i] 
             hs[1][:,:,j,:] = -1 * H[:,:,i,:] * delta(a,b) 
             hs[1][:,:,a,j] = -2 * H[:,:,i,b] 
             hs[1][b,:,a,:] =  2 * H[j,:,i,:] 
             hs[1][b,:,a,:] =  2 * H[:,j,i,:] 
             hs[1][:,:,a,:] = -1 * H[:,:,:,b] * delta(j,i) 
             hs[1][:,:,a,:] = -1 * H[:,:,b,:] * delta(j,i) 
             hs[1][i,:,:,:] = -1 * H[j,:,:,:] * delta(a,b) 
             hs[1][i,:,:,:] = -1 * H[:,j,:,:] * delta(a,b) 
          if(len(Emu)==2 and len(Enu)==4):
             a,i     = Emu
             b,c,k,j = Enu
             hs[2][c,i,:,j,k,:] =  2 * H[a,:,b,:] 
             hs[2][b,i,:,j,k,:] =  2 * H[a,:,c,:] 
             hs[2][c,i,:,j,k,:] =  2 * H[a,:,:,b] 
             hs[2][b,i,:,j,k,:] =  2 * H[a,:,:,c] 
             hs[2][b,c,:,j,:,:] = -1 * H[a,:,:,:] * delta(k,i) 
             hs[2][b,c,i,j,:,:] = -2 * H[a,k,:,:] 
             hs[2][b,c,:,k,:,:] = -1 * H[a,:,:,:] * delta(j,i) 
             hs[2][b,c,i,k,:,:] = -2 * H[a,j,:,:] 
             hs[2][c,i,:,j,k,:] =  2 * H[:,a,b,:] 
             hs[2][b,i,:,j,k,:] =  2 * H[:,a,c,:] 
             hs[2][c,i,:,j,k,:] =  2 * H[:,a,:,b] 
             hs[2][b,i,:,j,k,:] =  2 * H[:,a,:,c] 
             hs[2][b,c,:,j,:,:] = -1 * H[:,a,:,:] * delta(k,i) 
             hs[2][b,c,i,j,:,:] = -2 * H[k,a,:,:] 
             hs[2][b,c,:,k,:,:] = -1 * H[:,a,:,:] * delta(j,i) 
             hs[2][b,c,i,k,:,:] = -2 * H[j,a,:,:] 
             hs[2][c,:,:,j,k,:] = -1 * H[:,:,:,i] * delta(a,b) 
             hs[2][b,:,:,j,k,:] = -1 * H[:,:,:,i] * delta(a,c) 
             hs[2][c,:,:,a,j,k] = -2 * H[:,:,b,i] 
             hs[2][b,:,:,a,j,k] = -2 * H[:,:,c,i] 
             hs[2][b,c,:,a,j,:] =  2 * H[k,:,:,i] 
             hs[2][b,c,:,a,j,:] =  2 * H[:,k,:,i] 
             hs[2][b,c,:,a,k,:] =  2 * H[j,:,:,i] 
             hs[2][b,c,:,a,k,:] =  2 * H[:,j,:,i] 
             hs[2][c,:,:,j,k,:] = -1 * H[:,:,i,:] * delta(a,b) 
             hs[2][b,:,:,j,k,:] = -1 * H[:,:,i,:] * delta(a,c) 
             hs[2][c,:,:,a,j,k] = -2 * H[:,:,i,b] 
             hs[2][b,:,:,a,j,k] = -2 * H[:,:,i,c] 
             hs[2][b,c,:,a,j,:] =  2 * H[k,:,i,:] 
             hs[2][b,c,:,a,j,:] =  2 * H[:,k,i,:] 
             hs[2][b,c,:,a,k,:] =  2 * H[j,:,i,:] 
             hs[2][b,c,:,a,k,:] =  2 * H[:,j,i,:] 
             hs[2][i,:,:,j,k,:] =  1 * H[:,:,:,b] * delta(a,c) 
             hs[2][c,:,:,a,k,:] = -1 * H[:,:,:,b] * delta(j,i) 
             hs[2][c,:,:,a,j,:] = -1 * H[:,:,:,b] * delta(k,i) 
             hs[2][i,:,:,j,k,:] =  1 * H[:,:,:,c] * delta(a,b) 
             hs[2][b,:,:,a,k,:] = -1 * H[:,:,:,c] * delta(j,i) 
             hs[2][b,:,:,a,j,:] = -1 * H[:,:,:,c] * delta(k,i) 
             hs[2][i,:,:,j,k,:] =  1 * H[:,:,b,:] * delta(a,c) 
             hs[2][c,:,:,a,k,:] = -1 * H[:,:,b,:] * delta(j,i) 
             hs[2][c,:,:,a,j,:] = -1 * H[:,:,b,:] * delta(k,i) 
             hs[2][i,:,:,j,k,:] =  1 * H[:,:,c,:] * delta(a,b) 
             hs[2][b,:,:,a,k,:] = -1 * H[:,:,c,:] * delta(j,i) 
             hs[2][b,:,:,a,j,:] = -1 * H[:,:,c,:] * delta(k,i) 
             hs[2][c,i,:,j,:,:] = -1 * H[k,:,:,:] * delta(a,b) 
             hs[2][b,i,:,j,:,:] = -1 * H[k,:,:,:] * delta(a,c) 
             hs[2][b,c,:,a,:,:] =  1 * H[k,:,:,:] * delta(j,i) 
             hs[2][c,i,:,j,:,:] = -1 * H[:,k,:,:] * delta(a,b) 
             hs[2][b,i,:,j,:,:] = -1 * H[:,k,:,:] * delta(a,c) 
             hs[2][b,c,:,a,:,:] =  1 * H[:,k,:,:] * delta(j,i) 
             hs[2][c,i,:,k,:,:] = -1 * H[j,:,:,:] * delta(a,b) 
             hs[2][b,i,:,k,:,:] = -1 * H[j,:,:,:] * delta(a,c) 
             hs[2][b,c,:,a,:,:] =  1 * H[j,:,:,:] * delta(k,i) 
             hs[2][c,i,:,k,:,:] = -1 * H[:,j,:,:] * delta(a,b) 
             hs[2][b,i,:,k,:,:] = -1 * H[:,j,:,:] * delta(a,c) 
             hs[2][b,c,:,a,:,:] =  1 * H[:,j,:,:] * delta(k,i) 
          if(len(Emu)==4 and len(Enu)==2):
             a,b,j,i = Emu
             c,k     = Enu
          if(len(Emu)==4 and len(Enu)==4):
             a,b,j,i = Emu
             c,d,l,k = Enu

             hs[3][d,i,j,:,k,l,:,:] =  1 * H[b,:,:,:] * delta(a,c) 
             hs[3][c,i,j,:,k,l,:,:] =  1 * H[b,:,:,:] * delta(a,d) 
             hs[3][d,i,j,:,a,k,l,:] =  2 * H[b,:,c,:] 
             hs[3][c,i,j,:,a,k,l,:] =  2 * H[b,:,d,:] 
             hs[3][d,i,j,:,a,k,l,:] =  2 * H[b,:,:,c] 
             hs[3][c,i,j,:,a,k,l,:] =  2 * H[b,:,:,d] 
             hs[3][c,d,j,:,a,k,:,:] = -1 * H[b,:,:,:] * delta(l,i) 
             hs[3][c,d,i,:,a,k,:,:] = -1 * H[b,:,:,:] * delta(l,j) 
             hs[3][c,d,i,j,a,k,:,:] = -2 * H[b,l,:,:] 
             hs[3][c,d,j,:,a,l,:,:] = -1 * H[b,:,:,:] * delta(k,i) 
             hs[3][c,d,i,:,a,l,:,:] = -1 * H[b,:,:,:] * delta(k,j) 
             hs[3][c,d,i,j,a,l,:,:] = -2 * H[b,k,:,:] 
             hs[3][d,i,j,:,k,l,:,:] =  1 * H[:,b,:,:] * delta(a,c) 
             hs[3][c,i,j,:,k,l,:,:] =  1 * H[:,b,:,:] * delta(a,d) 
             hs[3][d,i,j,:,a,k,l,:] =  2 * H[:,b,c,:] 
             hs[3][c,i,j,:,a,k,l,:] =  2 * H[:,b,d,:] 
             hs[3][d,i,j,:,a,k,l,:] =  2 * H[:,b,:,c] 
             hs[3][c,i,j,:,a,k,l,:] =  2 * H[:,b,:,d] 
             hs[3][c,d,j,:,a,k,:,:] = -1 * H[:,b,:,:] * delta(l,i) 
             hs[3][c,d,i,:,a,k,:,:] = -1 * H[:,b,:,:] * delta(l,j) 
             hs[3][c,d,i,j,a,k,:,:] = -2 * H[l,b,:,:] 
             hs[3][c,d,j,:,a,l,:,:] = -1 * H[:,b,:,:] * delta(k,i) 
             hs[3][c,d,i,:,a,l,:,:] = -1 * H[:,b,:,:] * delta(k,j) 
             hs[3][c,d,i,j,a,l,:,:] = -2 * H[k,b,:,:] 
             hs[3][d,i,j,:,k,l,:,:] =  1 * H[a,:,:,:] * delta(b,c) 
             hs[3][c,i,j,:,k,l,:,:] =  1 * H[a,:,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,k,l,:] =  2 * H[a,:,c,:] 
             hs[3][c,i,j,:,b,k,l,:] =  2 * H[a,:,d,:] 
             hs[3][d,i,j,:,b,k,l,:] =  2 * H[a,:,:,c] 
             hs[3][c,i,j,:,b,k,l,:] =  2 * H[a,:,:,d] 
             hs[3][c,d,j,:,b,k,:,:] = -1 * H[a,:,:,:] * delta(l,i) 
             hs[3][c,d,i,:,b,k,:,:] = -1 * H[a,:,:,:] * delta(l,j) 
             hs[3][c,d,i,j,b,k,:,:] = -2 * H[a,l,:,:] 
             hs[3][c,d,j,:,b,l,:,:] = -1 * H[a,:,:,:] * delta(k,i) 
             hs[3][c,d,i,:,b,l,:,:] = -1 * H[a,:,:,:] * delta(k,j) 
             hs[3][c,d,i,j,b,l,:,:] = -2 * H[a,k,:,:] 
             hs[3][d,i,j,:,k,l,:,:] =  1 * H[:,a,:,:] * delta(b,c) 
             hs[3][c,i,j,:,k,l,:,:] =  1 * H[:,a,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,k,l,:] =  2 * H[:,a,c,:] 
             hs[3][c,i,j,:,b,k,l,:] =  2 * H[:,a,d,:] 
             hs[3][d,i,j,:,b,k,l,:] =  2 * H[:,a,:,c] 
             hs[3][c,i,j,:,b,k,l,:] =  2 * H[:,a,:,d] 
             hs[3][c,d,j,:,b,k,:,:] = -1 * H[:,a,:,:] * delta(l,i) 
             hs[3][c,d,i,:,b,k,:,:] = -1 * H[:,a,:,:] * delta(l,j) 
             hs[3][c,d,i,j,b,k,:,:] = -2 * H[l,a,:,:] 
             hs[3][c,d,j,:,b,l,:,:] = -1 * H[:,a,:,:] * delta(k,i) 
             hs[3][c,d,i,:,b,l,:,:] = -1 * H[:,a,:,:] * delta(k,j) 
             hs[3][c,d,i,j,b,l,:,:] = -2 * H[k,a,:,:] 
             hs[3][d,j,:,:,b,k,l,:] = -1 * H[:,:,:,i] * delta(a,c) 
             hs[3][c,j,:,:,b,k,l,:] = -1 * H[:,:,:,i] * delta(a,d) 
             hs[3][d,j,:,:,a,k,l,:] = -1 * H[:,:,:,i] * delta(b,c) 
             hs[3][c,j,:,:,a,k,l,:] = -1 * H[:,:,:,i] * delta(b,d) 
             hs[3][d,j,:,:,a,b,k,l] = -2 * H[:,:,c,i] 
             hs[3][c,j,:,:,a,b,k,l] = -2 * H[:,:,d,i] 
             hs[3][c,d,:,:,a,b,k,:] =  1 * H[:,:,:,i] * delta(l,j) 
             hs[3][c,d,j,:,a,b,k,:] =  2 * H[l,:,:,i] 
             hs[3][c,d,j,:,a,b,k,:] =  2 * H[:,l,:,i] 
             hs[3][c,d,:,:,a,b,l,:] =  1 * H[:,:,:,i] * delta(k,j) 
             hs[3][c,d,j,:,a,b,l,:] =  2 * H[k,:,:,i] 
             hs[3][c,d,j,:,a,b,l,:] =  2 * H[:,k,:,i] 
             hs[3][d,i,:,:,b,k,l,:] = -1 * H[:,:,:,j] * delta(a,c) 
             hs[3][c,i,:,:,b,k,l,:] = -1 * H[:,:,:,j] * delta(a,d) 
             hs[3][d,i,:,:,a,k,l,:] = -1 * H[:,:,:,j] * delta(b,c) 
             hs[3][c,i,:,:,a,k,l,:] = -1 * H[:,:,:,j] * delta(b,d) 
             hs[3][d,i,:,:,a,b,k,l] = -2 * H[:,:,c,j] 
             hs[3][c,i,:,:,a,b,k,l] = -2 * H[:,:,d,j] 
             hs[3][c,d,:,:,a,b,k,:] =  1 * H[:,:,:,j] * delta(l,i) 
             hs[3][c,d,i,:,a,b,k,:] =  2 * H[l,:,:,j] 
             hs[3][c,d,i,:,a,b,k,:] =  2 * H[:,l,:,j] 
             hs[3][c,d,:,:,a,b,l,:] =  1 * H[:,:,:,j] * delta(k,i) 
             hs[3][c,d,i,:,a,b,l,:] =  2 * H[k,:,:,j] 
             hs[3][c,d,i,:,a,b,l,:] =  2 * H[:,k,:,j] 
             hs[3][d,j,:,:,b,k,l,:] = -1 * H[:,:,i,:] * delta(a,c) 
             hs[3][c,j,:,:,b,k,l,:] = -1 * H[:,:,i,:] * delta(a,d) 
             hs[3][d,j,:,:,a,k,l,:] = -1 * H[:,:,i,:] * delta(b,c) 
             hs[3][c,j,:,:,a,k,l,:] = -1 * H[:,:,i,:] * delta(b,d) 
             hs[3][d,j,:,:,a,b,k,l] = -2 * H[:,:,i,c] 
             hs[3][c,j,:,:,a,b,k,l] = -2 * H[:,:,i,d] 
             hs[3][c,d,:,:,a,b,k,:] =  1 * H[:,:,i,:] * delta(l,j) 
             hs[3][c,d,j,:,a,b,k,:] =  2 * H[l,:,i,:] 
             hs[3][c,d,j,:,a,b,k,:] =  2 * H[:,l,i,:] 
             hs[3][c,d,:,:,a,b,l,:] =  1 * H[:,:,i,:] * delta(k,j) 
             hs[3][c,d,j,:,a,b,l,:] =  2 * H[k,:,i,:] 
             hs[3][c,d,j,:,a,b,l,:] =  2 * H[:,k,i,:] 
             hs[3][d,i,:,:,b,k,l,:] = -1 * H[:,:,j,:] * delta(a,c) 
             hs[3][c,i,:,:,b,k,l,:] = -1 * H[:,:,j,:] * delta(a,d) 
             hs[3][d,i,:,:,a,k,l,:] = -1 * H[:,:,j,:] * delta(b,c) 
             hs[3][c,i,:,:,a,k,l,:] = -1 * H[:,:,j,:] * delta(b,d) 
             hs[3][d,i,:,:,a,b,k,l] = -2 * H[:,:,j,c] 
             hs[3][c,i,:,:,a,b,k,l] = -2 * H[:,:,j,d] 
             hs[3][c,d,:,:,a,b,k,:] =  1 * H[:,:,j,:] * delta(l,i) 
             hs[3][c,d,i,:,a,b,k,:] =  2 * H[l,:,j,:] 
             hs[3][c,d,i,:,a,b,k,:] =  2 * H[:,l,j,:] 
             hs[3][c,d,:,:,a,b,l,:] =  1 * H[:,:,j,:] * delta(k,i) 
             hs[3][c,d,i,:,a,b,l,:] =  2 * H[k,:,j,:] 
             hs[3][c,d,i,:,a,b,l,:] =  2 * H[:,k,j,:] 
             hs[3][i,j,:,:,a,k,l,:] =  1 * H[:,:,:,c] * delta(b,d) 
             hs[3][i,j,:,:,b,k,l,:] =  1 * H[:,:,:,c] * delta(a,d) 
             hs[3][d,j,:,:,a,b,l,:] = -1 * H[:,:,:,c] * delta(k,i) 
             hs[3][d,i,:,:,a,b,l,:] = -1 * H[:,:,:,c] * delta(k,j) 
             hs[3][d,j,:,:,a,b,k,:] = -1 * H[:,:,:,c] * delta(l,i) 
             hs[3][d,i,:,:,a,b,k,:] = -1 * H[:,:,:,c] * delta(l,j) 
             hs[3][i,j,:,:,a,k,l,:] =  1 * H[:,:,:,d] * delta(b,c) 
             hs[3][i,j,:,:,b,k,l,:] =  1 * H[:,:,:,d] * delta(a,c) 
             hs[3][c,j,:,:,a,b,l,:] = -1 * H[:,:,:,d] * delta(k,i) 
             hs[3][c,i,:,:,a,b,l,:] = -1 * H[:,:,:,d] * delta(k,j) 
             hs[3][c,j,:,:,a,b,k,:] = -1 * H[:,:,:,d] * delta(l,i) 
             hs[3][c,i,:,:,a,b,k,:] = -1 * H[:,:,:,d] * delta(l,j) 
             hs[3][i,j,:,:,a,k,l,:] =  1 * H[:,:,c,:] * delta(b,d) 
             hs[3][i,j,:,:,b,k,l,:] =  1 * H[:,:,c,:] * delta(a,d) 
             hs[3][d,j,:,:,a,b,l,:] = -1 * H[:,:,c,:] * delta(k,i) 
             hs[3][d,i,:,:,a,b,l,:] = -1 * H[:,:,c,:] * delta(k,j) 
             hs[3][d,j,:,:,a,b,k,:] = -1 * H[:,:,c,:] * delta(l,i) 
             hs[3][d,i,:,:,a,b,k,:] = -1 * H[:,:,c,:] * delta(l,j) 
             hs[3][i,j,:,:,a,k,l,:] =  1 * H[:,:,d,:] * delta(b,c) 
             hs[3][i,j,:,:,b,k,l,:] =  1 * H[:,:,d,:] * delta(a,c) 
             hs[3][c,j,:,:,a,b,l,:] = -1 * H[:,:,d,:] * delta(k,i) 
             hs[3][c,i,:,:,a,b,l,:] = -1 * H[:,:,d,:] * delta(k,j) 
             hs[3][c,j,:,:,a,b,k,:] = -1 * H[:,:,d,:] * delta(l,i) 
             hs[3][c,i,:,:,a,b,k,:] = -1 * H[:,:,d,:] * delta(l,j) 
             hs[3][d,i,j,:,a,k,:,:] = -1 * H[l,:,:,:] * delta(b,c) 
             hs[3][c,i,j,:,a,k,:,:] = -1 * H[l,:,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,k,:,:] = -1 * H[l,:,:,:] * delta(a,c) 
             hs[3][c,i,j,:,b,k,:,:] = -1 * H[l,:,:,:] * delta(a,d) 
             hs[3][c,d,j,:,a,b,:,:] =  1 * H[l,:,:,:] * delta(k,i) 
             hs[3][c,d,i,:,a,b,:,:] =  1 * H[l,:,:,:] * delta(k,j) 
             hs[3][d,i,j,:,a,k,:,:] = -1 * H[:,l,:,:] * delta(b,c) 
             hs[3][c,i,j,:,a,k,:,:] = -1 * H[:,l,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,k,:,:] = -1 * H[:,l,:,:] * delta(a,c) 
             hs[3][c,i,j,:,b,k,:,:] = -1 * H[:,l,:,:] * delta(a,d) 
             hs[3][c,d,j,:,a,b,:,:] =  1 * H[:,l,:,:] * delta(k,i) 
             hs[3][c,d,i,:,a,b,:,:] =  1 * H[:,l,:,:] * delta(k,j) 
             hs[3][d,i,j,:,a,l,:,:] = -1 * H[k,:,:,:] * delta(b,c) 
             hs[3][c,i,j,:,a,l,:,:] = -1 * H[k,:,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,l,:,:] = -1 * H[k,:,:,:] * delta(a,c) 
             hs[3][c,i,j,:,b,l,:,:] = -1 * H[k,:,:,:] * delta(a,d) 
             hs[3][c,d,j,:,a,b,:,:] =  1 * H[k,:,:,:] * delta(l,i) 
             hs[3][c,d,i,:,a,b,:,:] =  1 * H[k,:,:,:] * delta(l,j) 
             hs[3][d,i,j,:,a,l,:,:] = -1 * H[:,k,:,:] * delta(b,c) 
             hs[3][c,i,j,:,a,l,:,:] = -1 * H[:,k,:,:] * delta(b,d) 
             hs[3][d,i,j,:,b,l,:,:] = -1 * H[:,k,:,:] * delta(a,c) 
             hs[3][c,i,j,:,b,l,:,:] = -1 * H[:,k,:,:] * delta(a,d) 
             hs[3][c,d,j,:,a,b,:,:] =  1 * H[:,k,:,:] * delta(l,i) 
             hs[3][c,d,i,:,a,b,:,:] =  1 * H[:,k,:,:] * delta(l,j) 
          return FermionicOperatorNBody(hs)

      def nested_commutator_adj_adj(self,Emu,Enu):
          hs = [np.zeros(tuple([self.n]*s)) for s in [2,4,6,8]]
          H  = self.htot[:,:,:,:]

          if(len(Emu)==2 and len(Enu)==2):
             a,i = Emu
             b,j = Enu
             hs[1][i,:,b,:] =  2 * H[a,:,j,:] 
             hs[1][i,:,b,:] =  2 * H[a,:,:,j] 
             hs[1][i,j,:,:] = -2 * H[a,b,:,:] 
             hs[1][i,:,b,:] =  2 * H[:,a,j,:] 
             hs[1][i,:,b,:] =  2 * H[:,a,:,j] 
             hs[1][i,j,:,:] = -2 * H[b,a,:,:] 
             hs[1][:,:,a,b] = -2 * H[:,:,j,i] 
             hs[1][j,:,a,:] =  2 * H[b,:,:,i] 
             hs[1][j,:,a,:] =  2 * H[:,b,:,i] 
             hs[1][:,:,a,b] = -2 * H[:,:,i,j] 
             hs[1][j,:,a,:] =  2 * H[b,:,i,:] 
             hs[1][j,:,a,:] =  2 * H[:,b,i,:] 
          if(len(Emu)==2 and len(Enu)==4):
             a,i     = Emu
             b,c,k,j = Enu
             hs[2][i,j,:,a,k,:] =  2 * H[b,:,k,:] 
             hs[2][i,j,:,a,k,:] =  2 * H[b,:,:,k] 
             hs[2][j,k,:,a,:,:] = -1 * H[b,:,:,:] * delta(k,i) 
             hs[2][i,k,:,a,:,:] = -1 * H[b,:,:,:] * delta(k,j) 
             hs[2][i,j,k,a,:,:] = -2 * H[b,k,:,:] 
             hs[2][i,j,:,a,k,:] =  2 * H[:,b,k,:] 
             hs[2][i,j,:,a,k,:] =  2 * H[:,b,:,k] 
             hs[2][j,k,:,a,:,:] = -1 * H[:,b,:,:] * delta(k,i) 
             hs[2][i,k,:,a,:,:] = -1 * H[:,b,:,:] * delta(k,j) 
             hs[2][i,j,k,a,:,:] = -2 * H[k,b,:,:] 
             hs[2][i,j,:,b,k,:] =  2 * H[a,:,k,:] 
             hs[2][i,j,:,b,k,:] =  2 * H[a,:,:,k] 
             hs[2][j,k,:,b,:,:] = -1 * H[a,:,:,:] * delta(k,i) 
             hs[2][i,k,:,b,:,:] = -1 * H[a,:,:,:] * delta(k,j) 
             hs[2][i,j,k,b,:,:] = -2 * H[a,k,:,:] 
             hs[2][i,j,:,b,k,:] =  2 * H[:,a,k,:] 
             hs[2][i,j,:,b,k,:] =  2 * H[:,a,:,k] 
             hs[2][j,k,:,b,:,:] = -1 * H[:,a,:,:] * delta(k,i) 
             hs[2][i,k,:,b,:,:] = -1 * H[:,a,:,:] * delta(k,j) 
             hs[2][i,j,k,b,:,:] = -2 * H[k,a,:,:] 
             hs[2][j,:,:,a,b,k] = -2 * H[:,:,k,i] 
             hs[2][k,:,:,a,b,:] =  1 * H[:,:,:,i] * delta(k,j) 
             hs[2][j,k,:,a,b,:] =  2 * H[k,:,:,i] 
             hs[2][j,k,:,a,b,:] =  2 * H[:,k,:,i] 
             hs[2][i,:,:,a,b,k] = -2 * H[:,:,k,j] 
             hs[2][k,:,:,a,b,:] =  1 * H[:,:,:,j] * delta(k,i) 
             hs[2][i,k,:,a,b,:] =  2 * H[k,:,:,j] 
             hs[2][i,k,:,a,b,:] =  2 * H[:,k,:,j] 
             hs[2][j,:,:,a,b,k] = -2 * H[:,:,i,k] 
             hs[2][k,:,:,a,b,:] =  1 * H[:,:,i,:] * delta(k,j) 
             hs[2][j,k,:,a,b,:] =  2 * H[k,:,i,:] 
             hs[2][j,k,:,a,b,:] =  2 * H[:,k,i,:] 
             hs[2][i,:,:,a,b,k] = -2 * H[:,:,j,k] 
             hs[2][k,:,:,a,b,:] =  1 * H[:,:,j,:] * delta(k,i) 
             hs[2][i,k,:,a,b,:] =  2 * H[k,:,j,:] 
             hs[2][i,k,:,a,b,:] =  2 * H[:,k,j,:] 
             hs[2][j,:,:,a,b,:] = -1 * H[:,:,:,i] 
             hs[2][i,:,:,a,b,:] = -1 * H[:,:,:,j] 
             hs[2][j,:,:,a,b,:] = -1 * H[:,:,i,:] 
             hs[2][i,:,:,a,b,:] = -1 * H[:,:,j,:] 
             hs[2][i,j,:,a,:,:] = -1 * H[b,:,:,:] 
             hs[2][i,j,:,b,:,:] = -1 * H[a,:,:,:] 
             hs[2][i,j,:,a,:,:] = -1 * H[:,b,:,:] 
             hs[2][i,j,:,b,:,:] = -1 * H[:,a,:,:]
          if(len(Emu)==4 and len(Enu)==2):
             a,b,j,i = Emu
             c,k     = Enu
          if(len(Emu)==4 and len(Enu)==4):
             a,b,j,i = Emu
             c,d,l,k = Enu
             hs[3][a,b,d,:,i,k,l,:] =  2 * H[j,:,c,:] 
             hs[3][a,b,c,:,i,k,l,:] =  2 * H[j,:,d,:] 
             hs[3][a,b,d,:,i,k,l,:] =  2 * H[j,:,:,c] 
             hs[3][a,b,c,:,i,k,l,:] =  2 * H[j,:,:,d] 
             hs[3][a,b,c,d,i,k,:,:] = -2 * H[j,l,:,:] 
             hs[3][a,b,c,d,i,l,:,:] = -2 * H[j,k,:,:] 
             hs[3][a,b,d,:,i,k,l,:] =  2 * H[:,j,c,:] 
             hs[3][a,b,c,:,i,k,l,:] =  2 * H[:,j,d,:] 
             hs[3][a,b,d,:,i,k,l,:] =  2 * H[:,j,:,c] 
             hs[3][a,b,c,:,i,k,l,:] =  2 * H[:,j,:,d] 
             hs[3][a,b,c,d,i,k,:,:] = -2 * H[l,j,:,:] 
             hs[3][a,b,c,d,i,l,:,:] = -2 * H[k,j,:,:] 
             hs[3][a,b,d,:,j,k,l,:] =  2 * H[i,:,c,:] 
             hs[3][a,b,c,:,j,k,l,:] =  2 * H[i,:,d,:] 
             hs[3][a,b,d,:,j,k,l,:] =  2 * H[i,:,:,c] 
             hs[3][a,b,c,:,j,k,l,:] =  2 * H[i,:,:,d] 
             hs[3][a,b,c,d,j,k,:,:] = -2 * H[i,l,:,:] 
             hs[3][a,b,c,d,j,l,:,:] = -2 * H[i,k,:,:] 
             hs[3][a,b,d,:,j,k,l,:] =  2 * H[:,i,c,:] 
             hs[3][a,b,c,:,j,k,l,:] =  2 * H[:,i,d,:] 
             hs[3][a,b,d,:,j,k,l,:] =  2 * H[:,i,:,c] 
             hs[3][a,b,c,:,j,k,l,:] =  2 * H[:,i,:,d] 
             hs[3][a,b,c,d,j,k,:,:] = -2 * H[l,i,:,:] 
             hs[3][a,b,c,d,j,l,:,:] = -2 * H[k,i,:,:] 
             hs[3][b,d,:,:,i,j,k,l] = -2 * H[:,:,c,a] 
             hs[3][b,c,:,:,i,j,k,l] = -2 * H[:,:,d,a] 
             hs[3][b,c,d,:,i,j,k,:] =  2 * H[l,:,:,a] 
             hs[3][b,c,d,:,i,j,k,:] =  2 * H[:,l,:,a] 
             hs[3][b,c,d,:,i,j,l,:] =  2 * H[k,:,:,a] 
             hs[3][b,c,d,:,i,j,l,:] =  2 * H[:,k,:,a] 
             hs[3][a,d,:,:,i,j,k,l] = -2 * H[:,:,c,b] 
             hs[3][a,c,:,:,i,j,k,l] = -2 * H[:,:,d,b] 
             hs[3][a,c,d,:,i,j,k,:] =  2 * H[l,:,:,b] 
             hs[3][a,c,d,:,i,j,k,:] =  2 * H[:,l,:,b] 
             hs[3][a,c,d,:,i,j,l,:] =  2 * H[k,:,:,b] 
             hs[3][a,c,d,:,i,j,l,:] =  2 * H[:,k,:,b] 
             hs[3][b,d,:,:,i,j,k,l] = -2 * H[:,:,a,c] 
             hs[3][b,c,:,:,i,j,k,l] = -2 * H[:,:,a,d] 
             hs[3][b,c,d,:,i,j,k,:] =  2 * H[l,:,a,:] 
             hs[3][b,c,d,:,i,j,k,:] =  2 * H[:,l,a,:] 
             hs[3][b,c,d,:,i,j,l,:] =  2 * H[k,:,a,:] 
             hs[3][b,c,d,:,i,j,l,:] =  2 * H[:,k,a,:] 
             hs[3][a,d,:,:,i,j,k,l] = -2 * H[:,:,b,c] 
             hs[3][a,c,:,:,i,j,k,l] = -2 * H[:,:,b,d] 
             hs[3][a,c,d,:,i,j,k,:] =  2 * H[l,:,b,:] 
             hs[3][a,c,d,:,i,j,k,:] =  2 * H[:,l,b,:] 
             hs[3][a,c,d,:,i,j,l,:] =  2 * H[k,:,b,:] 
             hs[3][a,c,d,:,i,j,l,:] =  2 * H[:,k,b,:]

          return FermionicOperatorNBody(hs)

      def build_overlap_operators(self):
          self.V_operators = []
          self.W_operators = []
          for mu in range(self.nexc):
              Emu = self.E_mu[mu]
              for nu in range(mu,self.nexc):
                  Enu = self.E_mu[nu]
                  print("START Overlap mu,nu ",mu,nu," ===== ")
                  V_mu_nu = self.commutator_adj_nor(Emu,Enu)
                  self.V_operators.append(V_mu_nu.mapping('jordan_wigner'))
                  print("END Overlap mu,nu ",mu,nu," ===== ")


      def build_hamiltonian_operators(self):
          self.M_operators = []
          self.Q_operators = []
          for mu in range(self.nexc):
              Emu = self.E_mu[mu]
              for nu in range(mu,self.nexc):
                  Enu = self.E_mu[nu]
                  M_mu_nu = self.nested_commutator_adj_nor(Emu,Enu)
                  Q_mu_nu = self.nested_commutator_adj_adj(Emu,Enu)
                  self.M_operators.append(M_mu_nu.mapping('jordan_wigner'))
                  self.Q_operators.append(Q_mu_nu.mapping('jordan_wigner'))
              print(mu,len(self.E_mu))

# ==========

n     = 12
nelec = 8
h1    = np.random.random((n,n))
h1    = (h1+h1.T)/2.0
h2    = np.random.random((n,n,3*n))
h2    = np.einsum('prg,qsg->prqs',h2,h2)

my_q_EOM = q_EOM(n,nelec,h1,h2)
my_q_EOM.build_excitation_operators()
my_q_EOM.build_overlap_operators()
#my_q_EOM.build_hamiltonian_operators()


