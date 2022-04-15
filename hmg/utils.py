"""
[References]
- https://stackoverflow.com/questions/8505651/non-repetitive-random-number-in-numpy
- https://algorithmist.com/wiki/Modular_inverse
- https://stackoverflow.com/questions/16044553/solving-a-modular-equation-python
"""

import random
import numpy as np


def get_bitwidth(n):
    """Calculate the bit width (size) for a given number of data elements
    """
    return int(2**np.ceil(np.log2(np.log2(2*n))))

def get_bytewidth(n):
    """Calculate the byte width (size) for a given number of data elements
    """
    return int(get_bitwidth(n) // 8)

def get_rand_indices(n_total, n_sample):
    return random.sample(range(n_total), n_sample)

def rand_degree_power(n_nodes, max_deg, alpha, dtype=np.uint32):
    rescaled = max_deg*(1 - np.random.power(alpha, size=n_nodes))
    arr = np.ceil(rescaled).astype(dtype)
    arr[::-1].sort()
    return arr

def iterative_egcd(a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q,r = b//a,b%a; m,n = x-u*q,y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y

def modinv(a, m):
    g, x, y = iterative_egcd(a, m) 
    if g != 1:
        return None
    else:
        return x % m
    
