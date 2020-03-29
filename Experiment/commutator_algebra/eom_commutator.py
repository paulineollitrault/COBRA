import numpy as np
import sys
sys.path.append('./src/')
from commutator import *

Bd = Operator_String_List(lst=[Operator_String(c=1,t=['Yai'],  o=['Ci','Da'],          s='fermi')])
H  = Operator_String_List(lst=[Operator_String(c=1,t=['Vprqs'],o=['Cp','Cq','Ds','Dr'],s='fermi')])
K  = Operator_String_List(lst=[Operator_String(c=1,t=['Xbk'],  o=['Cb','Dk'],          s='fermi')])


CBdH   = Bd.Commutator(H)
CBdHK1 = CBdH.Commutator(K)

CHK    = H.Commutator(K)
CBdHK2 = Bd.Commutator(CHK)

C = CBdHK1.Add(CBdHK2)
C.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
C.Print()

nb = 14
V  = np.random.random((nb,nb,nb,nb))
X  = np.random.random((nb,nb))
Y  = np.random.random((nb,nb))

Cm = matrix_elements(C,{'V':V,'X':X,'Y':Y},nb)

