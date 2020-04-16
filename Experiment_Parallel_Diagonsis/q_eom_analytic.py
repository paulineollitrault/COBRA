from utils import *

import time
import numpy as np
import itertools
from qiskit.aqua.operators import Z2Symmetries
import functools

class qEOM:

      def __init__(self,mol,psi,n,nelec,rank=0,size=1,logfile=None):

          self.molecule    = mol
          self.wfn_circuit = psi
          self.n           = n
          self.no          = sum(nelec)
          self.nv          = n-sum(nelec)
          self.occ    = [x for x in range(nelec[0])] + [n//2+x for x in range(nelec[1])]
          self.vrt     = list(set([x for x in range(n)])-set(self.occ))

          self.rank        = rank
          self.size        = size
          self.logfile     = logfile

          self.nexc        = 0
          self.elements    = []

          self.ne = np.eye(n)
          self.sz = np.eye(n)
          self.sz[n//2:,n//2:] = -self.sz[n//2:,n//2:]
          self.h1 = mol.mo_onee_ints
          self.h2 = mol.mo_eri_ints
          self.h  = np.zeros((n,n,n,n))
          t       = 0.5*self.h2+np.einsum('pr,qs->prqs',self.h1,np.eye(n//2)/(sum(nelec)-1))
          self.h[n//2:,n//2:,n//2:,n//2:] = t
          self.h[n//2:,:n//2,:n//2,n//2:] = t
          self.h[:n//2,n//2:,n//2:,:n//2] = t
          self.h[:n//2,:n//2,:n//2,:n//2] = t
          self.ss = np.zeros((n,n,n,n))
          sx = np.zeros((2,2))
          sy = np.zeros((2,2))
          sz = np.zeros((2,2))
          sx[0,1]=sx[1,0]=sy[1,0]=sz[0,0]=0.5
          sy[0,1]=sz[1,1]=-0.5
          spin_squared = np.einsum('mn,rh->mnrh',sx,sx)-np.einsum('mn,rh->mnrh',sy,sy)+np.einsum('mn,rh->mnrh',sz,sz)
          for i,j in itertools.product(range(n//2),repeat=2):
              for mu,nu,rho,eta in itertools.product(range(2),repeat=4):
                  self.ss[i+mu*n//2,j+rho*n//2,i+nu*n//2,j+eta*n//2] = spin_squared[mu,nu,rho,eta]
          self.ss += np.einsum('ij,kl->iklj',0.25*np.eye(n),np.eye(n)/(sum(nelec)-1))

          self.operators = {}
          self.time_statistics = {}

      # ==========================================================

      def build_excitation_operators(self):
          E1 = [(a,i) for a in self.vrt[:self.nv//2] for i in self.occ[:self.no//2]] \
             + [(a,i) for a in self.vrt[self.nv//2:] for i in self.occ[self.no//2:]]
          E2 = [(a,b,i,j) for a,b in itertools.product(self.vrt[:self.nv//2],repeat=2)              for i,j in itertools.product(self.occ[:self.no//2],repeat=2) if i<j and a<b] \
             + [(a,b,i,j) for a,b in itertools.product(self.vrt[self.nv//2:],repeat=2)              for i,j in itertools.product(self.occ[self.no//2:],repeat=2) if i<j and a<b] \
             + [(a,b,i,j) for a,b in itertools.product(self.vrt[:self.no//2],self.vrt[self.no//2:]) for i,j in itertools.product(self.occ[:self.no//2],self.occ[self.no//2:])]
          self.E_mu = E1+E2
         
          self.nexc        = len(self.E_mu)
          matrix_elements  = [(I,J) for I,J in itertools.product(range(self.nexc),repeat=2) if I<=J]
          self.elements    = [matrix_elements[self.rank+k*self.size] for k in range(len(matrix_elements)) if self.rank+k*self.size<len(matrix_elements)]

          self.logfile.write("Excitation number %d\n" % self.nexc)
          for mu,E in enumerate(self.E_mu):
              if(len(E)==2): self.logfile.write("Excitation %d --- %d %d\n" % (mu,E[0],E[1]))
              else:          self.logfile.write("Excitation %d --- %d %d %d %d\n" % (mu,E[0],E[1],E[2],E[3]))
          self.logfile.write("Total matrix elements %d\n" % len(matrix_elements))
          for (I,J) in self.elements:
              self.logfile.write("Matrix element connecting excitations %d and %d\n" % (I,J))

      def build_overlap_operators(self,qubit_mapping,two_qubit_reduction,epsilon,num_particles):
          self.V_elements = []
          self.W_elements = []
          self.times_VW   = []
          for (I,J) in self.elements:
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              VIJ_oper,VIJ_idx = commutator_adj_nor(self.n,EI,EJ)
              WIJ_oper,WIJ_idx = [None]*4,[None]*4
              t1 = time.time()
              self.logfile.write("V,W elements %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              VIJ_oper = VIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=VIJ_idx)
              if qubit_mapping.value == 'parity' and two_qubit_reduction:
                 VIJ_oper = Z2Symmetries.two_qubit_reduction(VIJ_oper,num_particles)
              self.V_elements.append(VIJ_oper)
              t2 = time.time()
              self.times_VW.append((I,J,t1-t0,t2-t1))
              self.logfile.write("V,W elements %d %d --- time for Mapping %f\n" % (I,J,t2-t1))

      def build_diagnosis_onebody_operators(self,qubit_mapping,two_qubit_reduction,epsilon,num_particles):
          self.M_ne_elements = []
          self.Q_ne_elements = []
          self.times_MQ_ne   = []

          for (I,J) in self.elements:
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              MIJ_oper,MIJ_idx = triple_commutator_adj_onebody_nor(self.n,EI,EJ,self.ne)
              QIJ_oper,QIJ_idx = triple_commutator_adj_onebody_adj(self.n,EI,EJ,self.ne)
              t1 = time.time()
              self.logfile.write("M,Q elements (ne) %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              MIJ_oper = MIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=MIJ_idx)
              QIJ_oper = QIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=QIJ_idx)
              if qubit_mapping == 'parity' and two_qubit_reduction:
                 MIJ_oper = Z2Symmetries.two_qubit_reduction(MIJ_oper,num_particles)
                 QIJ_oper = Z2Symmetries.two_qubit_reduction(QIJ_oper,num_particles)
              self.M_ne_elements.append(MIJ_oper)
              self.Q_ne_elements.append(QIJ_oper)
              t2 = time.time()
              self.times_MQ_ne.append((I,J,t1-t0,t2-t1))
              self.logfile.write("M,Q elements (ne) %d %d --- time for Mapping %f\n" % (I,J,t2-t1))

          self.M_sz_elements = []
          self.Q_sz_elements = []
          self.times_MQ_sz   = []

          for (I,J) in self.elements:
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              MIJ_oper,MIJ_idx = triple_commutator_adj_onebody_nor(self.n,EI,EJ,self.sz)
              QIJ_oper,QIJ_idx = triple_commutator_adj_onebody_adj(self.n,EI,EJ,self.sz)
              t1 = time.time()
              self.logfile.write("M,Q elements (sz) %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              MIJ_oper = MIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=MIJ_idx)
              QIJ_oper = QIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=QIJ_idx)
              if qubit_mapping == 'parity' and two_qubit_reduction:
                 MIJ_oper = Z2Symmetries.two_qubit_reduction(MIJ_oper,num_particles)
                 QIJ_oper = Z2Symmetries.two_qubit_reduction(QIJ_oper,num_particles)
              self.M_sz_elements.append(MIJ_oper)
              self.Q_sz_elements.append(QIJ_oper)
              t2 = time.time()
              self.times_MQ_sz.append((I,J,t1-t0,t2-t1))
              self.logfile.write("M,Q elements (sz) %d %d --- time for Mapping %f\n" % (I,J,t2-t1))

      def build_hamiltonian_operators(self,qubit_mapping,two_qubit_reduction,epsilon,num_particles):
          self.M_h_elements = []
          self.Q_h_elements = []
          self.times_MQ_h   = []

          for (I,J) in self.elements:
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              MIJ_oper,MIJ_idx = triple_commutator_adj_twobody_nor(self.n,EI,EJ,self.h)
              QIJ_oper,QIJ_idx = triple_commutator_adj_twobody_adj(self.n,EI,EJ,self.h)
              t1 = time.time()
              self.logfile.write("M,Q elements (H) %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              MIJ_oper = MIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=MIJ_idx)
              QIJ_oper = QIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=QIJ_idx)
              if qubit_mapping == 'parity' and two_qubit_reduction:
                 MIJ_oper = Z2Symmetries.two_qubit_reduction(MIJ_oper,num_particles)
                 QIJ_oper = Z2Symmetries.two_qubit_reduction(QIJ_oper,num_particles)
              self.M_h_elements.append(MIJ_oper)
              self.Q_h_elements.append(QIJ_oper)
              t2 = time.time()
              self.times_MQ_h.append((I,J,t1-t0,t2-t1))
              self.logfile.write("M,Q elements (H) %d %d --- time for Mapping %f\n" % (I,J,t2-t1))

      def build_diagnosis_twobody_operators(self,qubit_mapping,two_qubit_reduction,epsilon,num_particles):
          self.M_ss_elements = []
          self.Q_ss_elements = []
          self.times_MQ_ss   = []

          for (I,J) in self.elements:
              t0 = time.time()
              EI,EJ = self.E_mu[I],self.E_mu[J]
              MIJ_oper,MIJ_idx = triple_commutator_adj_twobody_nor(self.n,EI,EJ,self.ss)
              QIJ_oper,QIJ_idx = triple_commutator_adj_twobody_adj(self.n,EI,EJ,self.ss)
              t1 = time.time()
              self.logfile.write("M,Q elements (ss) %d %d --- time for FermionicOperator %f\n" % (I,J,t1-t0))
              MIJ_oper = MIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=MIJ_idx)
              QIJ_oper = QIJ_oper.mapping(map_type=qubit_mapping.value,threshold=epsilon,idx=QIJ_idx)
              if qubit_mapping == 'parity' and two_qubit_reduction:
                 MIJ_oper = Z2Symmetries.two_qubit_reduction(MIJ_oper,num_particles)
                 QIJ_oper = Z2Symmetries.two_qubit_reduction(QIJ_oper,num_particles)
              self.M_ss_elements.append(MIJ_oper)
              self.Q_ss_elements.append(QIJ_oper)
              t2 = time.time()
              self.times_MQ_ne.append((I,J,t1-t0,t2-t1))
              self.logfile.write("M,Q elements (ss) %d %d --- time for Mapping %f\n" % (I,J,t2-t1))

      # =====

      def run(self,quantum_instance):
          matrices = np.zeros((self.nexc,self.nexc,10))
          errors   = np.zeros((self.nexc,self.nexc,10))
          circuits = []
          for idx,VIJ_oper in enumerate(self.V_elements):
              if(not VIJ_oper.is_empty()):
                 circuit = VIJ_oper.construct_evaluation_circuit(
                           wave_function=self.wfn_circuit.construct_circuit(),
                           statevector_mode=quantum_instance.is_statevector,
                           use_simulator_snapshot_mode=False,
                           circuit_name_prefix=str(idx))
                 circuits.append(circuit)
          # =====
          if circuits:
              to_be_simulated_circuits = \
                  functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
              result = quantum_instance.execute(to_be_simulated_circuits)
          # ===== and output expectation values
          for idx,VIJ_oper in enumerate(self.V_elements):
              if(not VIJ_oper.is_empty()):
                 mean, std = VIJ_oper.evaluate_with_result(
                             result=result,statevector_mode=quantum_instance.is_statevector,
                             use_simulator_snapshot_mode=False,
                             circuit_name_prefix=str(idx))
                 I,J = self.elements[idx]
                 matrices[I,J,0],errors[I,J,0] = np.real(mean),np.abs(std)
                 matrices[J,I,0],errors[J,I,0] = np.real(mean),np.abs(std)
                 self.logfile.write("I,J,V[I,J] = %d %d %f +/- %f\n" % (I,J,matrices[I,J,0],errors[I,J,0]))
              else:
                 I,J = self.elements[idx]
                 self.logfile.write("I,J,V[I,J] = %d %d %f +/- %f\n" % (I,J,matrices[I,J,0],errors[I,J,0]))

      # =====

      def get_statistics(self,comm):

          comm.Barrier()

          if(self.rank==0):
              tvw    = np.zeros((self.nexc,self.nexc,2))
              tmq_ne = np.zeros((self.nexc,self.nexc,2))
              tmq_sz = np.zeros((self.nexc,self.nexc,2))
              tmq_h  = np.zeros((self.nexc,self.nexc,2))
              tmq_ss = np.zeros((self.nexc,self.nexc,2))
              tvw_list,tmq_ne_list,tmq_sz_list,tmq_h_list,tmq_ss_list = self.times_VW,self.times_MQ_ne,self.times_MQ_sz,self.times_MQ_h,self.times_MQ_ss
              for (I,J,t1,t2) in tvw_list:    tvw[I,J,:]    = [t1,t2]
              for (I,J,t1,t2) in tmq_ne_list: tmq_ne[I,J,:] = [t1,t2]
              for (I,J,t1,t2) in tmq_sz_list: tmq_sz[I,J,:] = [t1,t2]
              for (I,J,t1,t2) in tmq_h_list:  tmq_h[I,J,:]  = [t1,t2]
              for (I,J,t1,t2) in tmq_ss_list: tmq_ss[I,J,:] = [t1,t2]
          for k in range(1,self.size):
              comm.Barrier()
              if(self.rank==k):
                 comm.send([self.times_VW,self.times_MQ_ne,self.times_MQ_sz,self.times_MQ_h,self.times_MQ_ss],dest=0)
              if(self.rank==0):
                 tvw_list,tmq_ne_list,tmq_sz_list,tmq_h_list,tmq_ss_list = comm.recv(source=k)
                 for (I,J,t1,t2) in tvw_list:    tvw[I,J,:]    = [t1,t2]
                 for (I,J,t1,t2) in tmq_ne_list: tmq_ne[I,J,:] = [t1,t2]
                 for (I,J,t1,t2) in tmq_sz_list: tmq_sz[I,J,:] = [t1,t2]
                 for (I,J,t1,t2) in tmq_h_list:  tmq_h[I,J,:]  = [t1,t2]
                 for (I,J,t1,t2) in tmq_ss_list: tmq_ss[I,J,:] = [t1,t2]     
              comm.Barrier()

          comm.Barrier()

          if(self.rank==0):
             import matplotlib.pyplot as plt
             fig,axs = plt.subplots(2,2)
             tvw    = (tvw.transpose((1,0,2))+tvw)/2.0
             tmq_ne = (tmq_ne.transpose((1,0,2))+tmq_ne)/2.0 
             tmq_sz = (tmq_sz.transpose((1,0,2))+tmq_sz)/2.0
             tmq_h  = (tmq_h.transpose((1,0,2))+tmq_h)/2.0
             tmq_ss = (tmq_ss.transpose((1,0,2))+tmq_ss)/2.0
             d00 = tvw[:,:,0]+tvw[:,:,1]
             d01 = (tmq_ne[:,:,0]+tmq_ne[:,:,1]+tmq_sz[:,:,0]+tmq_sz[:,:,1])/2.0
             d10 = (tmq_h[:,:,0] +tmq_h[:,:,1] +tmq_ss[:,:,0]+tmq_ss[:,:,1])/2.0
             d11 = d00+d01+d10
             self.logfile.write("Total time per matrix element\n")
             self.logfile.write("Overlap  (FO) %f (MAP) %f\n" % (np.mean(tvw[:,:,0]),np.mean(tvw[:,:,1])))
             self.logfile.write("One-body (FO) %f (MAP) %f\n" % (np.mean(tmq_ne[:,:,0]+tmq_sz[:,:,0]),np.mean(tmq_ne[:,:,1]+tmq_sz[:,:,1])))
             self.logfile.write("Two-body (FO) %f (MAP) %f\n" % (np.mean(tmq_h[:,:,0]+tmq_ss[:,:,0]),np.mean(tmq_h[:,:,1]+tmq_ss[:,:,1])))

             vmin,vmax = d00.min(),d11.max()
             im = axs[0,0].imshow(d00,vmin=vmin,vmax=vmax)
             im = axs[0,1].imshow(d01,vmin=vmin,vmax=vmax)
             im = axs[1,0].imshow(d10,vmin=vmin,vmax=vmax)
             im = axs[1,1].imshow(d11,vmin=vmin,vmax=vmax)

             fig.subplots_adjust(right=0.8)
             cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
             fig.colorbar(im,cax=cbar_ax)
             plt.savefig('eom_timing.eps', format='eps')

