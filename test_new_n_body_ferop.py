import numpy as np
from qiskit.chemistry import FermionicOperatorNBody
from qiskit.chemistry import FermionicOperator
import time

nb = 12
h2 = np.random.random((nb,nb,nb,nb))
h1 = np.random.random((nb,nb))
#h2 = np.zeros((nb,nb,nb,nb))
#h1 = np.zeros((nb,nb))

#h2[0,1,2,3]=1

ferop_new = FermionicOperatorNBody([h1,h2])

tin = time.time()
qubitop_new = ferop_new.mapping('jordan_wigner')
tout = time.time()

print('execution time new = {}'.format(tout-tin))

ferop_old = FermionicOperator(h1,h2)

tin = time.time()
qubitop_old = ferop_old.mapping('jordan_wigner')
tout = time.time()

print('execution time old = {}'.format(tout-tin))

if qubitop_old == qubitop_new:
    print('CORRECT')
else:
    print('MISTAKE')
