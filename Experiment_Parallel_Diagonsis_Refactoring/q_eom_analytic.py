from utils import *
from qiskit.aqua.operators import Z2Symmetries

import time
import numpy as np
import itertools
import functools

class qEOM:

      def __init__(self,mol,psi,n,nelec,rank=0,size=1,comm=None,logfile=None):
          # ----- background information
          self.molecule    = mol
          self.wfn_circuit = psi
          self.n           = n
          self.no          = sum(nelec)
          self.nv          = n-sum(nelec)
          self.occ         = [x for x in range(nelec[0])] + [n//2+x for x in range(nelec[1])]
          self.vrt         = list(set([x for x in range(n)])-set(self.occ))
          self.nexc        = 0
          self.elements    = []

          # ----- hamiltonian
          self.h1 = mol.mo_onee_ints
          self.h2 = mol.mo_eri_ints
          self.h  = np.zeros((n,n,n,n))
          t1 = np.zeros((n,n))
          t1[:n//2,:n//2] = self.h1
          t1[n//2:,n//2:] = self.h1
          self.h  = np.einsum('pr,qs->prqs',t1,np.eye(n)/(sum(nelec)-1.0))
          self.h[:n//2,:n//2, :n//2,:n//2] += 0.5*self.h2
          self.h[:n//2,:n//2, n//2:,n//2:] += 0.5*self.h2
          self.h[n//2:,n//2:, :n//2,:n//2] += 0.5*self.h2
          self.h[n//2:,n//2:, n//2:,n//2:] += 0.5*self.h2
          # ----- diagnosis

          self.ne = np.eye(n)
          self.sz = np.eye(n)
          self.sz[n//2:,n//2:] = -self.sz[n//2:,n//2:]
          self.ss = np.zeros((n,n,n,n))
          sx = np.zeros((2,2))
          sy = np.zeros((2,2))
          sz = np.zeros((2,2))
          sx[0,1]=sx[1,0]=sy[1,0]=sz[0,0]=0.5
          sy[0,1]=sz[1,1]=-0.5
          spin_squared_tb = np.einsum('mn,rh->mnrh',sx,sx)-np.einsum('mn,rh->mnrh',sy,sy)+np.einsum('mn,rh->mnrh',sz,sz)
          spin_squared_ob = np.einsum('mnnh->mh',spin_squared_tb)
          for i in range(n//2):
              for mu,nu in itertools.product(range(2),repeat=2):
                  self.ss[i+mu*n//2,i+nu*n//2,:,:] = spin_squared_ob[mu,nu]*np.eye(n)/(sum(nelec)-1.0)
          for i,j in itertools.product(range(n//2),repeat=2):
              for mu,nu,rho,eta in itertools.product(range(2),repeat=4):
                  self.ss[i+mu*n//2,i+nu*n//2,j+rho*n//2,j+eta*n//2] += spin_squared_tb[mu,nu,rho,eta]

          # ----- parallelization and output 
          self.rank            = rank
          self.size            = size
          self.comm            = comm
          self.logfile         = logfile
          self.matrix_elements_adj_nor = {}
          self.matrix_elements_adj_adj = {}
          self.matrices_adj_nor        = {}
          self.matrices_adj_adj        = {}
          self.time_statistics         = {}

      # ==========================================================

      def build_excitation_operators(self):
          E1 = [(a,i) for a in self.vrt[:self.nv//2] for i in self.occ[:self.no//2]] \
             + [(a,i) for a in self.vrt[self.nv//2:] for i in self.occ[self.no//2:]]
          E2 = [(a,b,j,i) for a,b in itertools.product(self.vrt[:self.nv//2],repeat=2)              for i,j in itertools.product(self.occ[:self.no//2],repeat=2) if i<j and a<b] \
             + [(a,b,j,i) for a,b in itertools.product(self.vrt[self.nv//2:],repeat=2)              for i,j in itertools.product(self.occ[self.no//2:],repeat=2) if i<j and a<b] \
             + [(a,b,j,i) for a,b in itertools.product(self.vrt[:self.no//2],self.vrt[self.no//2:]) for i,j in itertools.product(self.occ[:self.no//2],self.occ[self.no//2:])]
          self.E_mu = E1+E2
         
          self.nexc        = len(self.E_mu)
          matrix_elements  = [(I,J) for I,J in itertools.product(range(self.nexc),repeat=2) if I<=J]
          self.elements    = [matrix_elements[self.rank+k*self.size] for k in range(len(matrix_elements)) if self.rank+k*self.size<len(matrix_elements)]

          self.logfile.write("Excitation number %d\n" % self.nexc)
          for mu,E in enumerate(self.E_mu):
              if(len(E)==2): self.logfile.write("Excitation %d --- %d %d\n"       % (mu,E[0],E[1]))
              else:          self.logfile.write("Excitation %d --- %d %d %d %d\n" % (mu,E[0],E[1],E[2],E[3]))
          self.logfile.write("Total matrix elements %d\n" % len(matrix_elements))
          for (I,J) in self.elements:
              self.logfile.write("Matrix element connecting excitations %d and %d\n" % (I,J))

      # ==========================================================

      def build_operators(self,task,qubit_mapping,two_qubit_reduction,num_particles,epsilon):
          self.matrix_elements_adj_nor[task] = []
          self.matrix_elements_adj_adj[task] = []
          self.time_statistics[task]         = []
          # -----
          for (I,J) in self.elements:
              # ----- computation
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              if(task=='overlap'):
                 adj_nor_oper,adj_nor_idx = commutator_adj_nor(self.n,EI,EJ)
                 adj_adj_oper,adj_adj_idx = commutator_adj_adj(self.n,EI,EJ)
              if(task=='hamiltonian'):
                 adj_nor_oper,adj_nor_idx = triple_commutator_adj_twobody_nor(self.n,EI,EJ,self.h)
                 adj_adj_oper,adj_adj_idx = triple_commutator_adj_twobody_adj(self.n,EI,EJ,self.h)
              if(task=='diagnosis_number'):
                 adj_nor_oper,adj_nor_idx = triple_commutator_adj_onebody_nor(self.n,EI,EJ,self.ne)
                 adj_adj_oper,adj_adj_idx = triple_commutator_adj_onebody_adj(self.n,EI,EJ,self.ne)
              if(task=='diagnosis_spin-z'):
                 adj_nor_oper,adj_nor_idx = triple_commutator_adj_onebody_nor(self.n,EI,EJ,self.sz)
                 adj_adj_oper,adj_adj_idx = triple_commutator_adj_onebody_adj(self.n,EI,EJ,self.sz)
              if(task=='diagnosis_spin-squared'):
                 adj_nor_oper,adj_nor_idx = triple_commutator_adj_twobody_nor(self.n,EI,EJ,self.ss)
                 adj_adj_oper,adj_adj_idx = triple_commutator_adj_twobody_adj(self.n,EI,EJ,self.ss)
              t1 = time.time()
              self.logfile.write("task: "+task+" elements %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              # ----- mapping
              adj_nor_oper = adj_nor_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=adj_nor_idx)
              adj_adj_oper = adj_adj_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=adj_adj_idx)
              if qubit_mapping.value == 'parity' and two_qubit_reduction:
                 adj_nor_oper = Z2Symmetries.two_qubit_reduction(adj_nor_oper,num_particles)
                 adj_adj_oper = Z2Symmetries.two_qubit_reduction(adj_adj_oper,num_particles)
              self.matrix_elements_adj_nor[task].append(adj_nor_oper)
              self.matrix_elements_adj_adj[task].append(adj_adj_oper)
              t2 = time.time()
              # ----- timings
              self.logfile.write("task: "+task+" elements %d %d --- time for Mapping %f\n" % (I,J,t2-t1))
              self.time_statistics[task].append((I,J,t1-t0,t2-t1))

      # ==========================================================
 
      def measure_matrices(self,quantum_instance,task,qubit_mapping,two_qubit_reduction,num_particles,epsilon):

          self.matrices_adj_nor[task],self.matrices_adj_adj[task] = np.zeros((self.nexc,self.nexc,2)),np.zeros((self.nexc,self.nexc,2))

          for (label,target,operator_list) in zip(['adj_nor','adj_adj'],[self.matrices_adj_nor[task],self.matrices_adj_adj[task]],
                                                  [self.matrix_elements_adj_nor[task],self.matrix_elements_adj_adj[task]]):
              circuits = []
              # -----
              for idx,oper in enumerate(operator_list):
                  if(not oper.is_empty()):
                     circuit = oper.construct_evaluation_circuit(wave_function=self.wfn_circuit.construct_circuit(),
                               statevector_mode=quantum_instance.is_statevector,use_simulator_snapshot_mode=False,circuit_name_prefix=str(idx))
                     circuits.append(circuit)
              # -----
              if circuits:
                  to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
                  result                   = quantum_instance.execute(to_be_simulated_circuits)
              # -----
              for idx,oper in enumerate(operator_list):
                  if(not oper.is_empty()):
                     mean,std = oper.evaluate_with_result(result=result,statevector_mode=quantum_instance.is_statevector,
                                use_simulator_snapshot_mode=False,circuit_name_prefix=str(idx))
                     mean,std = np.real(mean),np.abs(std)
                     I,J = self.elements[idx]
                     target[I,J,:] = mean,std
                     if(task!='overlap'):  target[I,J,:] /=  2.0
                     if(label=='adj_nor'): target[J,I,0]  =  target[I,J,0]; target[J,I,1] = target[I,J,1]
                     if(task=='overlap' and label=='adj_adj'): target[J,I,0]  = -target[I,J,0];          target[J,I,1] = target[I,J,1]
                     if(task!='overlap' and label=='adj_adj'): target[J,I,0]  =  target[I,J,0];          target[J,I,1] = target[I,J,1]
              for I,J in self.elements:
                     self.logfile.write("task: "+task+","+label+" --- I,J,x[I,J] = %d %d %f +/- %f\n" % (I,J,target[I,J,0],target[I,J,1]))
                     self.logfile.write("task: "+task+","+label+" --- I,J,x[I,J] = %d %d %f +/- %f\n" % (J,I,target[J,I,0],target[J,I,1]))

      # ==========================================================

      def run(self,quantum_instance,qubit_mapping,two_qubit_reduction,num_particles,epsilon):
          self.build_excitation_operators()
          for task in ['overlap','diagnosis_number','diagnosis_spin-z','hamiltonian','diagnosis_spin-squared']:
              self.build_operators(task,qubit_mapping,two_qubit_reduction,num_particles,epsilon)
              self.measure_matrices(quantum_instance,task,qubit_mapping,two_qubit_reduction,num_particles,epsilon)

          from mpi4py import MPI
          self.comm.Barrier()
          V_matrix = self.comm.reduce(self.matrices_adj_nor['overlap'],op=MPI.SUM,root=0)
          W_matrix = self.comm.reduce(self.matrices_adj_adj['overlap'],op=MPI.SUM,root=0)
          self.comm.Barrier()

          if(self.rank==0):
             for i in range(V_matrix.shape[0]):
                 for j in range(V_matrix.shape[1]):
                     self.logfile.write("overlap matrix %d %d %f,%f \n" % (i,j,V_matrix[i,j,0],W_matrix[i,j,0]))

             Overlap  = np.zeros((2*self.nexc,2*self.nexc))
   
             Overlap[:self.nexc,:self.nexc] =  V_matrix[:,:,0]
             Overlap[:self.nexc,self.nexc:] =  W_matrix[:,:,0]
             Overlap[self.nexc:,:self.nexc] = -W_matrix[:,:,0]
             Overlap[self.nexc:,self.nexc:] = -V_matrix[:,:,0]

          for task in ['hamiltonian','diagnosis_number','diagnosis_spin-z','diagnosis_spin-squared']:
              M_matrix = self.comm.reduce(self.matrices_adj_nor[task],op=MPI.SUM,root=0)
              Q_matrix = self.comm.reduce(self.matrices_adj_adj[task],op=MPI.SUM,root=0)
              self.comm.Barrier()

              if(self.rank==0):
    
                 for i in range(V_matrix.shape[0]):
                     for j in range(V_matrix.shape[1]):
                         self.logfile.write("%s matrix %d %d %f, %f \n" % (task,i,j,M_matrix[i,j,0],Q_matrix[i,j,0]))
       
                 Observable = np.zeros((2*self.nexc,2*self.nexc))
                 Observable[:self.nexc,:self.nexc] = M_matrix[:,:,0]
                 Observable[:self.nexc,self.nexc:] = Q_matrix[:,:,0]
                 Observable[self.nexc:,:self.nexc] = Q_matrix[:,:,0]
                 Observable[self.nexc:,self.nexc:] = M_matrix[:,:,0]
   
                 if(task=='hamiltonian'):
                      from scipy import linalg as LA
                      eps,U = LA.eig(Observable,Overlap)
                      for k,e in enumerate(eps):
                          self.logfile.write("eigenvalue %d: (%f,%f) \n" % (k,np.real(e),np.imag(e)))
                          for i in range(U.shape[0]):
                              self.logfile.write("   (%f,%f) \n" % (np.real(U[i,k]),np.imag(U[i,k])) ) 
 
                 for k in np.where(np.real(eps)>0)[0]:
                     zeta = np.einsum('i,ij,j',np.conj(U[:,k]),Observable,U[:,k])/np.einsum('i,ij,j',np.conj(U[:,k]),Overlap,U[:,k])
                     self.logfile.write("task: %s, eigenvector %d: (%f,%f) \n" % (task,k,np.real(zeta),np.imag(zeta)))

      # ==========================================================

      def get_statistics(self):

          if(self.rank==0):
             output  = np.zeros((self.nexc,self.nexc,2,2))

          for (r,c),task in zip([(0,0),(1,0),(0,1),(0,1),(1,0)],['overlap','hamiltonian','diagnosis_number','diagnosis_spin-z','diagnosis_spin-squared']):

              if(self.rank==0):
                time_matrix = np.zeros((self.nexc,self.nexc,2))
                for (I,J,t1,t2) in self.time_statistics[task]:
                    time_matrix[I,J,:] = [t1,t2]  
              self.comm.Barrier()

              for k in range(1,self.size):
                  if(self.rank==k):
                     self.comm.send(self.time_statistics[task],dest=0)
                  if(self.rank==0):
                     time_list = comm.recv(source=k)   
                     for (I,J,t1,t2) in time_list:
                         time_matrix[I,J,:] = [t1,t2]
                  self.comm.Barrier()
              
              if(self.rank==0):
                 time_matrix  = time_matrix+time_matrix.transpose((1,0,2))
                 time_matrix /= 2.0
                 
                 self.logfile.write("task: "+task+" --- total synthesis time: (FO) %f (MAP) %f\n" % (np.sum(time_matrix[:,:,0]),np.sum(time_matrix[:,:,1])))
                 output[:,:,r,c] += time_matrix[:,:,0]+time_matrix[:,:,1]
                 output[:,:,1,1] += time_matrix[:,:,0]+time_matrix[:,:,1]

          if(self.rank==0):
             import matplotlib.pyplot as plt
             fig,axs = plt.subplots(2,2)
             # -----
             vmin,vmax = output[:,:,0,0].min(),output[:,:,1,1].max()
             for r,c in itertools.product([0,1],repeat=2):
                 im = axs[r,c].imshow(output[:,:,r,c],vmin=vmin,vmax=vmax)
             # -----
             fig.subplots_adjust(right=0.8)
             cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
             fig.colorbar(im,cax=cbar_ax)
             plt.savefig('eom_timing.eps', format='eps')

