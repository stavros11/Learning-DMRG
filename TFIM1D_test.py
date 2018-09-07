# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:41:38 2018

@author: Admin
"""

import numpy as np
from dmrg import DMRG
pauli_X = np.array([[0., 1.], [1., 0.]])
pauli_Z = np.diag([1., -1.])

### Hamiltonian in MPO form ###
## DH: Hidden dimension for Hamiltonian
## H0: Hamiltonian for first site: Complex (DH, d, d)
## Hs: Hamiltonian for middle chain: Complex (N-2, DH, DH, d, d)
## HN: Hamiltonian for last site: Complex (DH, d, d)

def TFIM_MPO(N, h):
    ## Returns TFIM Hamiltonian in MPO Form
    ## N: number of sites
    ## h: field strength
    
    DH, d = 3, 2
    H0 = np.empty([DH, d, d])
    Hs = np.empty([DH, DH, d, d])
    HN = np.empty([DH, d, d])
    
    H0[0], HN[0] = h * pauli_X, np.eye(d)
    H0[1], HN[1] = pauli_Z, pauli_Z
    H0[2], HN[2] = np.eye(d), h * pauli_X
    
    Hs[0] = np.array([np.eye(d), np.zeros((d, d)), np.zeros((d, d))])
    Hs[1] = np.array([pauli_Z, np.zeros((d, d)), np.zeros((d, d))])
    Hs[2] = np.array([h * pauli_X, pauli_Z, np.eye(d)])
    
    return H0, Hs, HN


N = 6
H0, Hs, HN = TFIM_MPO(N=N, h=0.1)
Hs = np.array([Hs for i in range(N-2)])

## Bond dimensions (len=N-1)
D = [2, 3, 5, 3, 2]
x = DMRG(D=D, d=2, H0=H0, Hs=Hs, HN=HN, lcz_k=10)



