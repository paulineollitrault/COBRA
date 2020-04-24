"""
ucc_ec_driver.py
Handles the primary functions
"""

def run_ucc_r12(opt):
    """
    Parameters
    ----------
    * opt dictionary with user and default options/thresholds
    Returns
    -------
    * UCC-R12 ground state energy
        
    """

    import itertools
    import logging
    import numpy as np
    import time
    from datetime import datetime

    import qiskit    
    from qiskit                            import BasicAer,Aer    
    from qiskit.aqua                       import set_qiskit_aqua_logging, QuantumInstance
    from qiskit.aqua.operators             import Z2Symmetries, WeightedPauliOperator
    from qiskit.aqua.algorithms.adaptive   import VQE
    from qiskit.aqua.algorithms.classical  import ExactEigensolver
    from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,SLSQP, COBYLA

    from qiskit.chemistry.drivers                                      import PySCFDriver, UnitsType, HFMethodType
    from qiskit.chemistry.core                                         import TransformationType, QubitMappingType
    from qiskit.chemistry.aqua_extensions.components.initial_states    import HartreeFock
    from qiskit.chemistry                                              import set_qiskit_chemistry_logging

    # ----- local functions ----- #
    import sys
    sys.path.append('/Users/mario/Documents/GitHub/R12-F12/project_ec/ucc_ec/')
    from qmolecule_ec   import QMolecule_ec,build_parameter_list
    from hamiltonian_ec import Hamiltonian
    from uccsd_ec       import UCCSD

    from davidson import davidson

    #import logging
    #set_qiskit_chemistry_logging(logging.DEBUG)

    logfile = open(opt["logfile"],'w')
    dateTimeObj = datetime.now()
    logfile.write("date and time "+str(dateTimeObj)+"\n")
    logfile.write("\n\n")

    import subprocess
    label = subprocess.check_output(['git','log','-1','--format="%H"']).strip()
    logfile.write("git commit "+str(label)+"\n")
    logfile.write("\n\n")

    qiskit_dict = qiskit.__qiskit_version__
    logfile.write("qiskit version \n")
    for k in qiskit_dict:
    	logfile.write(k+" : "+qiskit_dict[k]+"\n")
    logfile.write("\n\n")

    logfile.write("local run options \n")
    for k in opt:
    	logfile.write(k+" : "+str(opt[k])+"\n")
    logfile.write("\n\n")

    import time
    t0 = time.time()

    if(opt["input_file"] is None):
     # ----- normal molecular calc ----- #
     if(opt["calc_type"].lower() == "rhf"):  hf=HFMethodType.RHF
     if(opt["calc_type"].lower() == "rohf"): hf=HFMethodType.ROHF
     if(opt["calc_type"].lower() == "uhf"):  hf=HFMethodType.UHF
     driver      = PySCFDriver(atom=opt["geometry"],unit=UnitsType.ANGSTROM,charge=opt["charge"],spin=opt["spin"],basis=opt["basis"],hf_method=hf)
     molecule    = driver.run()
     molecule_ec = QMolecule_ec(molecule=molecule,filename=None,logfile=logfile)
    else:
     # ----- custom matrix elements ----- #
     molecule_ec = QMolecule_ec(molecule=None,filename=opt["input_file"],calc=opt["calc_type"],nfreeze=opt["nfreeze"],logfile=logfile)

    molecule_ec.mo_eri_ints_ec = (molecule_ec.mo_eri_ints_ec).transpose((0,1,3,2))

    t1 = time.time()
    logfile.write("classical ES and setup: %f [s] \n" % (t1-t0))
    logfile.write("\n\n")

    core        = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,freeze_core=False)
    qubit_op, _ = core.run(molecule_ec)

    t2 = time.time()
    logfile.write("second-quantized Hamiltonian setup : %f [s] \n" % (t2-t1))
    logfile.write("\n\n")

    logfile.write("Original number of qubits %d \n" % (qubit_op.num_qubits))
    z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)

    nsym = len(z2_symmetries.sq_paulis)
    the_tapered_op = qubit_op
    sqlist = None
    z2syms = None

    if(nsym>0):

       logfile.write('\nZ2 symmetries found: \n')
       for symm in z2_symmetries.symmetries:
           logfile.write(symm.to_label())
       logfile.write('\nsingle-qubit operators found: \n')
       for sq in z2_symmetries.sq_paulis:
           logfile.write(sq.to_label())
       logfile.write('\nCliffords found: \n')
       for clifford in z2_symmetries.cliffords:
           logfile.write(clifford.print_details())
       logfile.write('\nsingle-qubit list: {} \n'.format(z2_symmetries.sq_list))
       
       tapered_ops = z2_symmetries.taper(qubit_op)
       for tapered_op in tapered_ops:
           logfile.write("Number of qubits of tapered qubit operator: %d \n" % (tapered_op.num_qubits))
       
       t3 = time.time()
       logfile.write("detection of symmetries: %f [s] \n" % (t3-t2))
     
       smallest_eig_value = 99999999999999
       smallest_idx = -1
      
       for idx in range(len(tapered_ops)):
           td0 = time.time()
           from utils import retrieve_SCF_energy
           print("In sector ",idx,len(tapered_ops))
           curr_value = retrieve_SCF_energy(tapered_ops[idx].copy(),core,opt) #,parm_list)
           curr_value = np.abs(curr_value-molecule_ec.hf_energy)
           print("Deviation ",curr_value)
           if curr_value < smallest_eig_value:
              smallest_eig_value = curr_value
              smallest_idx = idx
              if(curr_value<1e-6): break
           td1 = time.time()
           val = curr_value
           logfile.write("Lowest-energy computational basis state of the {}-th tapered operator is %s %f \n" % (str(idx),val))
           logfile.write("HF search time %f: \n" % (td1-td0))
     
       the_tapered_op = tapered_ops[smallest_idx]
       the_coeff = tapered_ops[smallest_idx].z2_symmetries.tapering_values
       logfile.write("{}-th tapered operator, with corresponding symmetry sector of {}".format(smallest_idx, the_coeff))
       logfile.write("\nNumber of qubits in the {}-th tapered operator {}\n\n".format(smallest_idx,the_tapered_op.num_qubits))

       sqlist = the_tapered_op.z2_symmetries.sq_list
       z2syms = the_tapered_op.z2_symmetries

    # ========
    # setup initial state
    # ========

    init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits, num_orbitals=core._molecule_info['num_orbitals'],
                        qubit_mapping=core._qubit_mapping, two_qubit_reduction=core._two_qubit_reduction,
                        num_particles=core._molecule_info['num_particles'],sq_list=sqlist)

    # ---- initial parameter guess 
    init_parm = None
    if(opt["start_pt"].lower()=="file"):
       init_parm = np.loadtxt('vqe.parameters')
    if(opt["var_form"].lower()=="uccsd" and opt["start_pt"].lower()=="ccsd"):
       parm_list    = build_parameter_list(molecule_ec)

    logfile.write("Initial parameters = %d\n" % len(parm_list))
    for mu,p in enumerate(parm_list):
        logfile.write('%d %f\n' % (mu,p))

    if(opt["var_form"].lower()=="uccsd"):
        var_form = UCCSD(num_qubits=the_tapered_op.num_qubits,depth=opt["UCCSD_depth"],
                   num_orbitals=core._molecule_info['num_orbitals'],
                   num_particles=core._molecule_info['num_particles'],
                   active_occupied=opt["UCCSD_active_occupied"], active_unoccupied=opt["UCCSD_active_unoccupied"], initial_state=init_state,
                   qubit_mapping=core._qubit_mapping, two_qubit_reduction=core._two_qubit_reduction,
                   num_time_slices=opt["UCCSD_num_time_slices"],z2_symmetries=z2syms,init_parm=parm_list)

        if(opt["var_form"].lower()=="uccsd" and opt["start_pt"].lower()=="ccsd"):
           nparm,ndepth = len(var_form._mask),var_form._depth
           init_parm    = np.zeros(nparm*ndepth)
           for idp in range(ndepth):
            for ims in range(nparm):
             init_parm[ims+idp*nparm] = parm_list[var_form._mask[ims]]

        logfile.write("Selected parameters = %d\n" % nparm)
        for mu,p in enumerate(var_form._mask):
            logfile.write('%d %f\n' % (p,parm_list[p]))

    elif(opt["var_form"].lower()=="ry"):
        var_form = RY(the_tapered_op.num_qubits,depth=opt["R_depth"],
                   entanglement=opt["R_entanglement"],initial_state=HF_state)
    elif(opt["var_form"].lower()=="ryrz"):
        var_form = RYRZ(the_tapered_op.num_qubits,depth=opt["R_depth"],
                   entanglement=opt["R_entanglement"],initial_state=HF_state)
    elif(opt["var_form"].lower()=="swaprz"):
        var_form = SwapRZ(the_tapered_op.num_qubits,depth=opt["R_depth"],
                   entanglement=opt["R_entanglement"],initial_state=HF_state)
    else:
        print("invalid variational form")
        assert(False)

    # setup optimizer
    if(  opt["optimizer"].lower()=="bfgs"):  optimizer = L_BFGS_B(maxiter=opt["max_eval"])
    elif(opt["optimizer"].lower()=="cg"):    optimizer = CG(maxiter=opt["max_eval"])
    elif(opt["optimizer"].lower()=="slsqp"): optimizer = SLSQP(maxiter=opt["max_eval"])
    elif(opt["optimizer"].lower()=="spsa"):  optimizer = SPSA()
    elif(opt["optimizer"].lower()=="cobyla"): optimizer = COBYLA(maxiter=opt["max_eval"])
    else:                                    print("not coded yet"); assert(False) 

    # set vqe
    if(opt["var_form"].lower()=="uccsd"): algo = VQE(the_tapered_op,var_form,optimizer,initial_point=init_parm)
    else:                                 algo = VQE(the_tapered_op,var_form,optimizer)
    # setup backend
    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend=backend)

    t0 = time.time()

    algo_result = algo.run(quantum_instance)

    t1 = time.time()

    logfile.write("\nVQE time [s] %f \n\n" % (t1-t0))

    result = core.process_algorithm_result(algo_result)
    for line in result[0]:
        logfile.write(line+"\n")
    
    logfile.write("\nThe parameters for UCCSD are:\n")
    for i,(tc,tq) in enumerate(zip(init_parm,algo_result['opt_params'])):
     logfile.write("%d %f %f \n" % (i,tc,tq))

    if(opt["print_parameters"]):
       par_file = open('vqe.parameters','w')
       for p in algo_result['opt_params']:
        par_file.write("%f \n" % p)
       par_file.close()

    #td0 = time.time()
    #ee = davidson(the_tapered_op,'fci')
    #td1 = time.time()
    #logfile.write("\n\nExact diagonalization, energy: %f \n" % (ee+molecule_ec.nuclear_repulsion_energy+molecule_ec.energy_offset_ec))

    #logfile.write("Davidson FCI time: %f [s] \n" % (td1-td0))
    
    print('============================================================================')
    print('                                      DONE!')
    print('============================================================================')
    

    return 0

