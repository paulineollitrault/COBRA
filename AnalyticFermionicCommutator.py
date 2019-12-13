#!/usr/bin/env python
# coding: utf-8


import itertools
import logging
import sys
import copy

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import aqua_globals

logger = logging.getLogger(__name__)

class term:
    #this class is for elemntary term of arbitrary order of the fermionic operator coef*adag_i*adag_k*a_l*a_j
    def __init__(self, coef, indlist):
        self._coef = coef # coefficient in fron of the fermionic product
        self._indlist = indlist # list of indicies of the ferminonic operators in the normal form
    
    @property        
    def coef(self):
        return self._coef

    def set_coef(self, new_coef):  
        self._coef = new_coef
        
    @property        
    def indlist(self):
        return self._indlist
    
    @property
    def order(self):
        return len(self._indlist)
    
def h_matrix_to_terms(h):
    res = []
    dim  = len(h.shape)
    size = h.shape[0]
    for x in itertools.product(range(size),repeat = dim):
        res.append(term(h[x],list(x)))
    return res

def scMult(scalar,oper,threshold=1e-12):
    res = copy.deepcopy(oper)
    if abs(scalar)<=threshold:
        return []
    else:
        for x in res:
            x.set_coef(x.coef*scalar)
    return res

def equal(st1,st2):
    return np.array_equal(np.array(st1.indlist),np.array(st2.indlist))

def smaller(st1,st2):
    if len(st1.indlist)<len(st2.indlist):
        return True
    elif (len(st1.indlist)==len(st2.indlist))and(equal(st1,st2)==False):
    #    return ((np.array(st2.indlist)-np.array(st1.indlist))>=0).all()
        tmp = np.array(st2.indlist)-np.array(st1.indlist)
        for i in range(len(tmp)):
            if tmp[i]>0:
                return True
            elif tmp[i]<0:
                return False
    else:
        return False   
    
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if smaller(x,pivot)]
    middle = [x for x in arr if equal(x,pivot)]
    right = [x for x in arr if smaller(pivot,x)]
    return quicksort(left) + middle + quicksort(right)

def SimplifyOper(oper,threshold=1e-12):
    res = copy.deepcopy(oper)
    Finished  = False
    while Finished==False:
        BoolA = []
        for i in range(len(res)-1):
            BoolA.append(equal(res[i],res[i+1]))
        if np.array(BoolA).any()==True:
            Finished  = False
        else:
            Finished  = True
        
        if Finished==False:
            for i in range(len(res)-1):
                if equal(res[i],res[i+1])==True:
                    new_coef = res[i].coef+res[i+1].coef
                    if abs(new_coef)<threshold:
                        #kill both
                        res.pop(i+1)
                        res.pop(i)
                    else:
                        #kill i+1
                        res[i].set_coef(new_coef)
                        res.pop(i)
                    break
    return res

def Addition(oper1,oper2,threshold=1e-12):
    res = copy.deepcopy(oper1)+copy.deepcopy(oper2)
    res = quicksort(res)
    res = SimplifyOper(res,threshold)
    return res

def Subtraction(oper1,oper2,threshold=1e-12):    
    res = copy.deepcopy(oper1)+scMult(-1,oper2,threshold)
    res = quicksort(res)
    res = SimplifyOper(res,threshold)
    return res

def quicksortL(arr):
    #quick sort just for arrays
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot ]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if pivot < x]
    return quicksortL(left) + middle + quicksortL(right)

def order(b):
    count = 0
    ideal = np.array(quicksortL(b))
    temp = b
    
    DoOrder = True
    
    while DoOrder==True:
        if np.array_equal(ideal,np.array(temp))==True:
            DoOrder = False

        for i in range(len(temp)-1):
            if temp[i]>temp[i+1]:
                tmp = temp[i]
                temp[i] = temp[i+1]
                temp[i+1] = tmp
                count +=1
                break
    return count


def TermMultiply(term1,term2,threshold=1e-12):
    c1 = term1.coef
    c2 = term2.coef
    
    l1 = term1.indlist
    l2 = term2.indlist
    
    #adag -> 0, a -> 1
    
    op = list(np.zeros(len(l1)//2))+list(np.ones(len(l1)//2))+list(np.zeros(len(l2)//2))+list(np.ones(len(l2)//2))
    
    unsort = [1,l1+l2,op]
    
    res = [unsort]
    
    Ordered = False
    forceBreak1 = False 
    
    newphase = 1
    newind = []
    newop = []
    UpdateRes = False
    
    while Ordered == False:
        if UpdateRes == True:
            res.append([newphase,newind,newop])
            UpdateRes = False

        BoolA = [np.array(x[2][:len(x[2])//2]).any()==1 for x in res]
        Test1 = np.array(BoolA).any()==True #if test1 is True need to do ordering
        
        Bool2 = [np.array_equal(np.array(x[1][:len(x[1])//2]),np.array(quicksortL(x[1][:len(x[1])//2]))) for x in res]
        Test2 = np.array(Bool2).any()==False #if test2 is True need to do ordering of creation indexies
        
        Bool3 = [np.array_equal(np.array(x[1][len(x[1])//2:]),np.array(quicksortL(x[1][len(x[1])//2:]))) for x in res]
        Test3 = np.array(Bool3).any()==False #if test3 is True need to do ordering of annih indexies
        
        if (Test1 == False)and(Test2==False)and(Test3==False):
            Ordered = True

        
        if Test1 == True:
            for x in res:
                forceBreak1 = False
                forceBreak2 = False
                if np.array(x[2][:len(x[2])//2]).any()==1:
                    #need to do a step of ordering
                    for i in range(len(x[2])-1):
                        if (x[2][i]==1)and(x[2][i+1]==0):
                            #do the swap
                            if x[1][i]!=x[1][i+1]:
                                #swap+phase
                                tmp = x[1][i]
                                x[1][i] = x[1][i+1]
                                x[1][i+1] = tmp
                                
                                x[2][i]=0
                                x[2][i+1] = 1
                                
                                x[0] *=-1
                                forceBreak1 = True
                                break
                            elif x[1][i]==x[1][i+1]:
                                #add list
                                newphase = copy.deepcopy(x[0])
                                
                                newind = copy.deepcopy(x[1])
                                newind.pop(i+1)
                                newind.pop(i)
                                
                                newop  = copy.deepcopy(x[2])
                                newop.pop(i+1)
                                newop.pop(i)
                                UpdateRes = True
                                                                
                                x[2][i]=0
                                x[2][i+1] = 1
                                
                                x[0] *=-1
                                #swap+phase
                                forceBreak2 = True
                                break
                if forceBreak1==True:
                    break
                    
                if forceBreak2==True:
                    break
                    
        if (Test1 == False)and((Test2 == True)or(Test3 == True)):
            for x in res:
                x_cr = x[1][:len(x[1])//2]
                x_an = x[1][len(x[1])//2:]
                phasefactor = 1
                if (order(x_cr)+order(x_an))%2==1:
                    phasefactor = -1
                x[0] *=phasefactor
                x[1] = quicksortL(x_cr)+quicksortL(x_an)
    realres = []
    for x in res:
        realres.append(term(c1*c2*x[0],x[1]))
    realres = quicksort(realres)
    realres = SimplifyOper(realres,threshold)
    return realres

def Multiply(oper1,oper2,threshold=1e-12):
    res = []
    o1 = quicksort(oper1)
    o1 = SimplifyOper(o1,threshold)

    o2 = quicksort(oper2)
    o2 = SimplifyOper(o2,threshold)
    
    for x in o1:
        for y in o2:
            res=Addition(res,TermMultiply(x,y,threshold),threshold)
    res = quicksort(res)
    res = SimplifyOper(res,threshold)    
    return res

def RemoveDoubles(oper,threshold=1e-12):
    o1 = copy.deepcopy(oper)
    o1 = quicksort(o1)
    o1 = SimplifyOper(o1,threshold)
    Removed = False
    while Removed == False:
        for i in range(len(o1)):
            ind = o1[i].indlist
            cr_ind = ind[:len(ind)//2]
            an_ind = ind[len(ind)//2:]
            
            np_cr_ind = np.array(cr_ind)
            np_an_ind = np.array(an_ind)
            cp_cr_ind = np.array(list(set(cr_ind)))
            cp_an_ind = np.array(list(set(an_ind)))
            if (np.array_equal(np_cr_ind,cp_cr_ind)==False)or(np.array_equal(np_an_ind,cp_an_ind)==False):
                o1.pop(i)
                break
            elif i == len(o1)-1:
                Removed = True
    return o1


def FermCommutator(oper1,oper2,threshold=1e-12):
    o1 = quicksort(oper1)
    o1 = SimplifyOper(o1,threshold)

    o2 = quicksort(oper2)
    o2 = SimplifyOper(o2,threshold)    
    
    ab = Multiply(o1,o2,threshold)
    ba = Multiply(o2,o1,threshold)
    res = Subtraction(ab,ba,threshold)
    res = quicksort(res)
    res = SimplifyOper(res,threshold)
    
    res = RemoveDoubles(res,threshold=1e-12)
    return res

def oper_to_h(oper,nferm):
    reslist = []
    for x in oper:
        reslist.append(int(x.order))
    reslist=quicksortL(list(set(reslist)))
    
    res = []
    for dim in reslist:
        res.append(np.zeros(np.full((dim,), nferm)))
        
    for x in oper:
        x_order = x.order
        x_coef  = x.coef
        x_ind   = x.indlist
        
        res[reslist.index(x_order)][tuple(x_ind)]=x_coef
    return res


# In[ ]:




