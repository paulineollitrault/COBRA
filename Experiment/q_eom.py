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
          self.h1    = h1[:,:]
          self.h2    = h2[:,:,:,:]
          self.htot  = FermionicOperator(h1,h2)

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
             b,c,j,k = Enu
             hs[1][c,i,j,k] =  delta(a,b)
             hs[1][b,i,j,k] =  delta(a,c)
             hs[1][b,c,a,j] = -delta(k,i)
             hs[1][b,c,a,k] = -delta(j,i)
          if(len(Emu)==4 and len(Enu)==2):
             a,b,i,j = Emu
             c,k     = Enu
             hs[1][i,j,a,k] =  delta(b,c)
             hs[1][i,j,b,k] =  delta(a,c)
             hs[1][c,j,a,b] = -delta(k,i)
             hs[1][c,i,a,b] = -delta(k,j)
          if(len(Emu)==4 and len(Enu)==4):
             a,b,i,j = Emu
             c,d,k,l = Enu
             hs[2][d,i,j,a,k,l] =  delta(b,c)
             hs[2][c,i,j,a,k,l] =  delta(b,d)
             hs[2][d,i,j,b,k,l] =  delta(a,c)
             hs[2][c,i,j,b,k,l] =  delta(a,d)
             hs[2][c,d,j,a,b,k] = -delta(l,i)
             hs[2][c,d,i,a,b,k] = -delta(l,j)
             hs[2][c,d,j,a,b,l] = -delta(k,i)
             hs[2][c,d,i,a,b,l] = -delta(k,j)
          return FermionicOperatorNBody(hs)

      def build_overlap_operators(self):
          self.V_operators = []
          self.W_operators = []
          for mu in range(self.nexc):
              Emu = self.E_mu[mu]
              for nu in range(mu,self.nexc):
                  Enu = self.E_mu[nu]
                  print("overlap ",self.E_mu[mu],mu,self.E_mu[nu],nu)
                  V_mu_nu = self.commutator_adj_nor(Emu,Enu)
                  self.V_operators.append(V_mu_nu)


#      def build_hamiltonian_operators(self):
#          self.M_operators = []
#          self.Q_operators = []
#          for mu in range(self.nexc):
#              oper_mu     = self.excitation_to_k_body(mu,dagger=False)
#              oper_mu_dag = self.excitation_to_k_body(mu,dagger=True)
#
#              for nu in range(mu,self.nexc):
#                  oper_nu     = self.excitation_to_k_body(nu,dagger=False)
#                  oper_nu_dag = self.excitation_to_k_body(nu,dagger=True)
#
#                  print("hamiltonian ",self.E_mu[mu],mu,self.E_mu[mu],nu)
#
#                  V_mu_nu = ten_commutator(oper_mu_dag,self.htot,oper_nu)
#                  W_mu_nu = ten_commutator(oper_mu_dag,self.htot,oper_nu_dag)
#
#                  self.M_operators.append(V_mu_nu)
#                  self.Q_operators.append(W_mu_nu)

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


