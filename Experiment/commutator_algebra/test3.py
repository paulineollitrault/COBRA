import sys
sys.path.append('./src/')
from commutator import *

X = Operator_String_List(lst=[Operator_String(c=1,t=['Hprqs'],o=['Cp','Cq','Ds','Dr'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ca','Cb','Di','Dj'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cc','Cd','Dk','Dl'],s='fermi')])

CXY = X.Commutator(Y)
CXY.Print()

C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
C_CXY_Z.Print()

