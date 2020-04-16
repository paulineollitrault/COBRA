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

#driver    = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.0 0.0 1.4; H 0.0 0.0 2.8; H 0.0 0.0 4.2; H 0.0 0.0 5.6; H 0.0 0.0 7.0;''',unit=UnitsType.BOHR,charge=0,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF)
driver    = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.0 0.0 1.4''',unit=UnitsType.BOHR,charge=0,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF)
#driver    = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.0 0.0 1.4; H 0.0 0.0 2.8; H 0.0 0.0 4.2''',unit=UnitsType.BOHR,charge=0,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF)
#driver    = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.0 0.0 1.4; H 0.0 0.0 2.8''',unit=UnitsType.BOHR,charge=0,spin=1,basis='sto-6g',hf_method=HFMethodType.ROHF)
#driver    = PySCFDriver(atom='''H 0.0 0.0 0.0; H 0.0 0.0 1.4; H 0.0 0.0 2.8; H 0.0 0.0 4.2; H 0.0 0.0 5.6''',unit=UnitsType.BOHR,charge=0,spin=1,basis='sto-6g',hf_method=HFMethodType.ROHF)

molecule  = driver.run()
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,freeze_core=False)
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

algo = qEOM(molecule,init_state,core._molecule_info['num_orbitals'],core._molecule_info['num_particles'],rank,size,outf)
algo.build_excitation_operators()
algo.build_overlap_operators(          qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,epsilon=1e-8,num_particles=core._molecule_info['num_particles'])
algo.build_diagnosis_onebody_operators(qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,epsilon=1e-8,num_particles=core._molecule_info['num_particles'])
algo.build_hamiltonian_operators(      qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,epsilon=1e-8,num_particles=core._molecule_info['num_particles'])
algo.build_diagnosis_twobody_operators(qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,epsilon=1e-8,num_particles=core._molecule_info['num_particles'])

backend = Aer.get_backend('statevector_simulator')
result  = algo.run(QuantumInstance(backend,seed_simulator=aqua_globals.random_seed,seed_transpiler=aqua_globals.random_seed))

algo.get_statistics(comm)




'''
var_form  = var_form_homolumo(qubit_op.num_qubits)
optimizer = COBYLA(maxiter=0) #L_BFGS_B(maxiter=400)

algo = VQE(qubit_op,var_form,optimizer,aux_operators=anc_op)

print(algo._operator.print_details())

backend = Aer.get_backend('statevector_simulator')
result  = algo.run(QuantumInstance(backend,seed_simulator=aqua_globals.random_seed,seed_transpiler=aqua_globals.random_seed))
print("shift core,ph,rep ",core._energy_shift,core._ph_energy_shift,core._nuclear_repulsion_energy)
print(result['min_val']+core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy)
print(result['opt_params'])

print(var_form.construct_circuit(result['opt_params']))

naux = result['aux_ops'].shape[1]
for i in range(naux):
    if(i==naux-1): result['aux_ops'][0,i,0] += (core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy)
    print(">>> aux ops ",result['aux_ops'][0,i,0],result['aux_ops'][0,i,1])

print(algo._operator.print_details())

for mu in range(2**qubit_op.num_qubits):
    print(mu,result['min_vector'][mu])
'''


'''
simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(2, 2)
x_rank = format(rank,'02b')
for m,x in enumerate(x_rank):
    if(x==str(1)): circuit.x(m)

circuit.h(0)
circuit.cx(0,1)
circuit.measure([0,1],[0,1])
job = execute(circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(circuit)

outf.write("Bell state "+str(x_rank)+"\n")
outf.write(str(circuit.draw()))
outf.write("\nTotal counts\n")
for k in counts.keys():
    outf.write("   %s %s\n" % (k,str(counts[k])))
'''


