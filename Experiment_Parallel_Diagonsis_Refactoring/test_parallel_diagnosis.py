#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

outf = open('log_'+str(rank),'w')
outf.write("Logfile of process %d of %d on %s.\n" % (rank,size,name))

from qiskit                                     import *
from qiskit.chemistry.drivers                   import PySCFDriver, UnitsType, HFMethodType
from qiskit.chemistry.core                      import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua                                import QuantumInstance,aqua_globals

nh  = 4
geo = ''
for ih in range(nh):
    geo += 'H 0.0 0.0 '+str(ih*1.0)+'; '
geo = geo[:len(geo)-2]

import pyscf
from pyscf import gto,scf,fci,ao2mo

mol = gto.M(atom=geo,charge=0,spin=0)
mf = scf.RHF(mol)
E = mf.kernel()

cisolver = fci.FCI(mol,mf.mo_coeff)
E,ci     = cisolver.kernel(nroots=6)

print("FCI excitation energies ")
print(E-min(E))

c = mf.mo_coeff

from functools import reduce
import numpy

h1 = reduce(numpy.dot,(c.T,mf.get_hcore(),c))
h2 = ao2mo.incore.full(mf._eri,c)

H_fci = fci.direct_spin1.pspace(h1,h2,mol.nao_nr(),mol.nelectron)[1]
print("diagonal of FCI matrix")
print(numpy.diag(H_fci)-H_fci[0,0])


'''
fs = fci.addons.fix_spin_(fci.FCI(mol, m.mo_coeff), .5)
fs.nroots = 3
e, c = fs.kernel(verbose=5)
for i, x in enumerate(c):
    print('state %d, E = %.12f  2S+1 = %.7f' %
          (i, e[i], fci.spin_op.spin_square0(x, norb, nelec)[1]))
'''

driver    = PySCFDriver(atom=geo,unit=UnitsType.ANGSTROM,charge=0,spin=0,basis='sto-3g',hf_method=HFMethodType.RHF)
molecule  = driver.run()
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.JORDAN_WIGNER,two_qubit_reduction=False,freeze_core=False)
H_op,A_op = core.run(molecule)

outf.write("HF energy %f\n" % molecule.hf_energy)
outf.write("Hamiltonian\n")
outf.write(H_op.print_details())

init_state = HartreeFock(num_qubits=H_op.num_qubits,num_orbitals=core._molecule_info['num_orbitals'],
                    qubit_mapping=core._qubit_mapping,two_qubit_reduction=core._two_qubit_reduction,
                    num_particles=core._molecule_info['num_particles'])

outf.write("State\n")
outf.write(str(init_state.construct_circuit().draw())+"\n")

from q_eom_analytic import *

algo    = qEOM(molecule,init_state,core._molecule_info['num_orbitals'],core._molecule_info['num_particles'],rank,size,comm,outf)
backend = Aer.get_backend('statevector_simulator')
result  = algo.run(QuantumInstance(backend,shots=1,seed_simulator=aqua_globals.random_seed,seed_transpiler=aqua_globals.random_seed),
                   qubit_mapping=QubitMappingType.JORDAN_WIGNER,two_qubit_reduction=False,num_particles=core._molecule_info['num_particles'],epsilon=1e-8)
algo.get_statistics()

