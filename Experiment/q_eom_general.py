import numpy as np
import time
from qiskit.chemistry import FermionicOperator
from TensorAnalyticFermionicCommutator import ten_commutator
from fermionic_operator_nbody import FermionicOperatorNBody
from utils import commutator_adj_nor,nested_commutator_adj_nor,nested_commutator_adj_adj

class q_EOM:

      def __init__(self,n,nelec,h1,h2,algorithm='cobra'):
          self.n     = n
          self.nelec = nelec
          self.htot  = FermionicOperator(h1,h2)
          self.algo  = algorithm
          self.hten  = 0.5*h2[:,:,:,:]+np.einsum('pr,qs->pqsr',h1,np.eye(n)/(self.nelec-1))

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

      def excitation_to_k_body_operator(self,mu,adjoint=False):
          hs = [np.zeros(tuple([self.n]*s)) for s in [2,4,6,8]]
          idx = self.E_mu[mu]
          if(adjoint): hs[len(idx)//2-1][tuple(idx[::-1])] = 1.0
          else:        hs[len(idx)//2-1][tuple(idx)] = 1.0
          return FermionicOperator(hs[0],hs[1])

      def build_overlap_operators(self,every=1):
          self.V_operators = []
          self.W_operators = []
          
          if(self.algo=='cobra'):
          
             for mu in range(0,self.nexc,every):
                 Emu_dag = self.excitation_to_k_body_operator(mu,adjoint=True)
                 for nu in range(mu,self.nexc,every):
                     t0 = time.time()
                     Enu = self.excitation_to_k_body_operator(nu)
                     V_mu_nu = ten_commutator(Emu_dag,Enu)
                     t1 = time.time()
                     self.V_operators.append(V_mu_nu.mapping('jordan_wigner'))
                     t2 = time.time()
   
                     Enu_dag = self.excitation_to_k_body_operator(nu,adjoint=True)
                     W_mu_nu = ten_commutator(Emu_dag,Enu_dag)
                     t3 = time.time()
                     self.W_operators.append(W_mu_nu.mapping('jordan_wigner'))
                     t4 = time.time()
   
                     print("COBRA - Overlap matrix_element ",mu,nu," times (V,W,comm,map) ",t1-t0,t2-t1,t3-t2,t4-t3)

          if(self.algo=='analytic'):

             for mu in range(0,self.nexc,every):
                 for nu in range(mu,self.nexc,every):
                     t0 = time.time()
                     V_mu_nu = commutator_adj_nor(self.n,self.E_mu[mu],self.E_mu[nu])
                     t1 = time.time()
                     self.V_operators.append(V_mu_nu.mapping('jordan_wigner'))
                     t2 = time.time()
 
                     W_mu_nu = None
                     t3 = time.time()
                     self.W_operators.append(None)
                     t4 = time.time()
   
                     print("ANALYTIC - Overlap matrix_element ",mu,nu," times (V,W,comm,map) ",t1-t0,t2-t1,t3-t2,t4-t3)

      def build_hamiltonian_operators(self,every=1):
          self.M_operators = []
          self.Q_operators = []

          if(self.algo=='cobra'):

             for mu in range(0,self.nexc,every):
                 Emu_dag = self.excitation_to_k_body_operator(mu,adjoint=True)
                 for nu in range(mu,self.nexc,every):
                     t0 = time.time()
                     Enu = self.excitation_to_k_body_operator(nu)
                     M_mu_nu = ten_commutator(Emu_dag,Enu,self.htot)
                     t1 = time.time()
                     self.M_operators.append(M_mu_nu.mapping('jordan_wigner'))
                     t2 = time.time()
   
                     Enu_dag = self.excitation_to_k_body_operator(nu,adjoint=True)
                     Q_mu_nu = ten_commutator(Emu_dag,Enu_dag,self.htot)
                     t3 = time.time()
                     self.Q_operators.append(Q_mu_nu.mapping('jordan_wigner'))
                     t4 = time.time()
   
                     print("COBRA - Hamilton matrix_element ",mu,nu," times (M,Q,comm,map) ",t1-t0,t2-t1,t3-t2,t4-t3)

          if(self.algo=='analytic'):

             for mu in range(0,self.nexc,every):
                 for nu in range(mu,self.nexc,every):
                     t0 = time.time()
                     M_mu_nu = nested_commutator_adj_nor(self.n,self.hten,self.E_mu[mu],self.E_mu[nu])
                     t1 = time.time()
                     self.M_operators.append(M_mu_nu.mapping('jordan_wigner'))
                     t2 = time.time()
   
                     Enu_dag = self.excitation_to_k_body_operator(nu,adjoint=True)
                     Q_mu_nu = nested_commutator_adj_adj(self.n,self.hten,self.E_mu[mu],self.E_mu[nu])
                     t3 = time.time()
                     self.Q_operators.append(Q_mu_nu.mapping('jordan_wigner'))
                     t4 = time.time()
   
                     print("ANALYTIC - Hamilton matrix_element ",mu,nu," times (M,Q,comm,map) ",t1-t0,t2-t1,t3-t2,t4-t3)


# ==========

#every_list=[1,4,8,10,20]
every_list=[10,10]

for m,n in enumerate([10,12]): #[4,6,8,10,12]):
 nelec = max(n//2,2)
 h1    = np.random.random((n,n))
 h1    = (h1+h1.T)/2.0
 h2    = np.random.random((n,n,3*n))
 h2    = np.einsum('prg,qsg->prqs',h2,h2)

 #my_q_EOM = q_EOM(n,nelec,h1,h2,algorithm='cobra')
 #my_q_EOM.build_excitation_operators()
 #ta = time.time()
 #print(">>> ",n,my_q_EOM.nexc)
 #my_q_EOM.build_overlap_operators(every=every_list[m])
 #tb = time.time()
 #print("total time ",tb-ta)
 #print(" ")
 #print(" ")
 #ta = time.time()
 #my_q_EOM.build_hamiltonian_operators(every=every_list[m])
 #tb = time.time()
 #print("total time ",tb-ta)
 #print(" ")
 #print(" ")

 print("==============================")

 my_q_EOM = q_EOM(n,nelec,h1,h2,algorithm='analytic')
 my_q_EOM.build_excitation_operators()
 ta = time.time()
 print(">>> ",n,my_q_EOM.nexc)
 my_q_EOM.build_overlap_operators(every=every_list[m])
 tb = time.time()
 print("total time ",tb-ta)
 print(" ")
 print(" ")
 ta = time.time()
 my_q_EOM.build_hamiltonian_operators(every=every_list[m])
 tb = time.time()
 print("total time ",tb-ta)
 print(" ")
 print(" ")

