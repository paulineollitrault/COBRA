import sys
sys.path.append('./src/')
from commutator import *

print("The hamiltonian matrix requires computing commutators:")
print("Between 1-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cb','Dj'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 1-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cj','Db'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("Between 1-body and 2-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cb','Cc','Dj','Dk'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 1-body and 2-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ci','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ck','Cj','Dc','Db'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 2-body and 1-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cc','Dk'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 1-body and 2-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Ck','Dc'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 2-body operators -- ADJ NOR")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cc','Cd','Dk','Dl'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

print("Between 1-body and 2-body operators -- ADJ ADJ")
X = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cj','Ci','Db','Da'],s='fermi')])
Y = Operator_String_List(lst=[Operator_String(c=1,t=['Hpqrs'],o=['Cp','Cq','Dr','Ds'],s='fermi')])
Z = Operator_String_List(lst=[Operator_String(c=1,t=[       ],o=['Cl','Ck','Dd','Dc'],s='fermi')])

CXY = X.Commutator(Y)
C_CXY_Z = CXY.Commutator(Z)
C_CXY_Z.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

CYZ = Y.Commutator(Z)
C_X_CYZ = X.Commutator(CYZ)
C_X_CYZ.Remove_OV_terms(['i','j','k','l'],['a','b','c','d'])

C = C_CXY_Z.Add(C_X_CYZ)
C.Simplify()
C.Print()

print("===========")

