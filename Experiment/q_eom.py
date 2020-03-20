import numpy as np
from qiskit.chemistry import FermionicOperator
from TensorAnalyticFermionicCommutator import ten_commutator

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

      def excitation_to_k_body(self,mu,dagger=False):
          excitation_mu = self.E_mu[mu]
          oper1 = np.zeros((self.n,self.n))
          oper2 = np.zeros((self.n,self.n,self.n,self.n))
          print(excitation_mu)
          if(len(excitation_mu)==2):
             a,i  = excitation_mu
             if(dagger): oper1[i,a] = 1.0
             else:       oper1[a,i] = 1.0
             return FermionicOperator(oper1,oper2)
          else:
             a,b,i,j = excitation_mu
             if(dagger): oper2[j,i,b,a] = 1.0
             else:       oper2[a,b,i,j] = 1.0
             return FermionicOperator(oper1,oper2)

      def build_overlap_operators(self):
          self.V_operators = []
          self.W_operators = []
          for mu in range(self.nexc):
              oper_mu     = self.excitation_to_k_body(mu,dagger=False)
              oper_mu_dag = self.excitation_to_k_body(mu,dagger=True)

              for nu in range(mu,self.nexc):
                  oper_nu     = self.excitation_to_k_body(nu,dagger=False)
                  oper_nu_dag = self.excitation_to_k_body(nu,dagger=True)

                  V_mu_nu = ten_commutator(oper_mu_dag,oper_nu)
                  W_mu_nu = ten_commutator(oper_mu_dag,oper_nu_dag)

                  self.V_operators.append(V_mu_nu)
                  self.W_operators.append(W_mu_nu)

      def build_hamiltonian_operators(self):
          self.M_operators = []
          self.Q_operators = []
          for mu in range(self.nexc):
              oper_mu     = self.excitation_to_k_body(mu,dagger=False)
              oper_mu_dag = self.excitation_to_k_body(mu,dagger=True)

              for nu in range(mu,self.nexc):
                  oper_nu     = self.excitation_to_k_body(nu,dagger=False)
                  oper_nu_dag = self.excitation_to_k_body(nu,dagger=True)

                  V_mu_nu = ten_commutator(oper_mu_dag,self.htot,oper_nu)
                  W_mu_nu = ten_commutator(oper_mu_dag,self.htot,oper_nu_dag)

                  self.M_operators.append(V_mu_nu)
                  self.Q_operators.append(W_mu_nu)

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
my_q_EOM.build_hamiltonian_operators()

