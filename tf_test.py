# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:15:15 2018

@author: Admin
"""

import numpy as np
from itertools import product
from TFIM1D_ED import Ham

N = 10
h = 1
eigvals, eigvecs = np.linalg.eigh(Ham)

psi = eigvecs[:,2]

def classical_energy(states, pbc=False):
    x = 1 - 2 * states
    energy = np.sum(x[:, 1:] * x[:, :-1], axis=1)
    if pbc:
        energy+= x[:, 0] * x[:, 1]
    return -energy

states = np.array([list(x) for x in product([0, 1], repeat=N)])
to_decimal = 2**np.arange(N-1, -1, -1)
cl = classical_energy(states)

en_theory = np.sum(psi * Ham.dot(psi))

cross_term = 0
for i in range(N):
    flipped_states = np.copy(states)
    flipped_states[:,i] = (states[:,i] == 0).astype(np.int)
    cross_term += np.sum(psi * psi[flipped_states.dot(to_decimal)])

en_mine = (np.sum(psi*psi*cl) - h * cross_term) / np.sum(psi*psi)

print(en_theory, en_mine)