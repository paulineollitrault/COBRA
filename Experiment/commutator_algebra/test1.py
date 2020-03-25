import sys
sys.path.append('./src/')
from commutator import *

X = Operator_String_List(lst=[Operator_String(c=1,t=['Hpq'],o=['Cp','Dq'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Tai'],o=['Ca','Di'],s='fermi')])

CXY = X.Commutator(Y)
CXY.Print()

