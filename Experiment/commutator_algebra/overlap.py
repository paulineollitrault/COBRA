import sys
sys.path.append('./src/')
from commutator import *

print("The overlap matrix requires computing commutators:")
print("Between 1-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cb','Dj'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j'],['a','b'])
CXY.Simplify()
CXY.Print()
print("===========")
print("Between 1-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cj','Db'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j'],['a','b'])
CXY.Simplify()
CXY.Print()
print("===========")

# ===================

print("Between 1-body and 2-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cb','Cc','Dj','Dk'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
CXY.Print()
print("===========")
print("Between 1-body and 2-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Ck','Cj','Dc','Db'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
CXY.Simplify()
CXY.Print()
print("===========")

# ===================

print("Between 2-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cc','Cd','Dk','Dl'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
CXY.Simplify()
CXY.Print()
print("===========")
print("Between 2-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=[],o=['Cl','Ck','Dc','Dd'],s='fermi')])
CXY = X.Commutator(Y)
CXY.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])
CXY.Simplify()
CXY.Print()


