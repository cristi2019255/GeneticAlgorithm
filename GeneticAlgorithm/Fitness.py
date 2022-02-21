import numpy as np

k = 4

def CO(x):
    return np.sum(x)

def B_1(x):
    # d = 1
    co = CO(x)
    if co == k:
        return k
    else:
        return k - 1 - co

def B_2(x):
    # d = 2.5
    co = CO(x)
    if co == k:
        return k
    else:
        return k - 2.5 - (k- 2.5)/(k-1) * co

def TF_deceptive_linked(x):
    l = len(x)
    m = int(l/k)    
    return np.sum([B_1(x[j*k : j*k + k]) for j in range(m)])


def TF_non_deceptive_linked(x):
    l = len(x)
    m = int(l/k)    
    return np.sum([B_2(x[j*k : j*k + k]) for j in range(m)])

def TF_deceptive_not_linked(x):
    l = len(x)
    m = int(l/k)    
    return np.sum([B_1(x[j : : m]) for j in range(m)])

def TF_non_deceptive_not_linked(x):
    l = len(x)
    m = int(l/k)    
    return np.sum([B_2(x[j : : m]) for j in range(m)])