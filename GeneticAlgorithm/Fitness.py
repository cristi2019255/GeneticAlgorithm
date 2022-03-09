import numpy as np
from numba import jit

l = 40
k = 4
m = int(l/k)    

@jit()
def CO(x):
    return np.sum(x)

@jit()
def B_1(x):
    # d = 1
    co = CO(x)
    if co == k:
        return k
    else:
        return k - 1 - co

@jit()
def B_2(x):
    # d = 2.5
    co = CO(x)
    if co == k:
        return k
    else:
        return k - 2.5 - (k- 2.5)/(k-1) * co

@jit()
def TF_deceptive_linked(x):        
    return np.sum(np.array([B_1(x[j*k : j*k + k]) for j in range(m)]))

@jit()
def TF_non_deceptive_linked(x):    
    return np.sum(np.array([B_2(x[j*k : j*k + k]) for j in range(m)]))

@jit()
def TF_deceptive_not_linked(x):    
    return np.sum(np.array([B_1(x[j : : m]) for j in range(m)]))

@jit()
def TF_non_deceptive_not_linked(x):    
    return np.sum(np.array([B_2(x[j : : m]) for j in range(m)]))

FITNESS_FUNCTIONS = [CO, TF_deceptive_linked, TF_deceptive_not_linked, TF_non_deceptive_linked, TF_non_deceptive_not_linked]