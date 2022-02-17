import numpy as np

k = 4
d = 1

def CO(x):
    return np.sum(x)

def B(x):
    co = CO(x)
    if co == k:
        return k
    else:
        return k - d - (k-d)/(k-1) * co

def TF(x):
    l = len(x)
    m = int(l/k)    
    return np.sum([B(x[j*k : j*k + k]) for j in range(m)])