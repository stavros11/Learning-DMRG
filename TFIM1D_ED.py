# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:16:19 2018

@author: Admin
"""

import numpy as np
pauli_X = np.array([[0., 1.], [1., 0.]])
pauli_Z = np.diag([1., -1.])

def kron_list(matrices_list):
    ## np.kron of a list of matrices
    term = np.kron(matrices_list[0], matrices_list[1])
    if len(matrices_list) < 3:
        return term
    else:
        return kron_list([term] + matrices_list[2:])
    
def create_hamiltonian(N=10, h=1.0, pbc=False):
    ### Create Hamiltonian Matrix ###
    Ham = np.zeros([2**N, 2**N])
    HB = np.zeros([2**N, 2**N])
    
    ## Add interactions to Hamiltonian
    interaction = [np.kron(pauli_Z, pauli_Z)]
    identities = [np.eye(2) for i in range(N-2)]
    
    for i in range(N-1):
        Ham += kron_list(identities[:i] + interaction + identities[i:])
    
    ## Add PBC term
    if pbc: 
        Ham += kron_list([pauli_Z] + identities + [pauli_Z])

    ## Add field interactions
    identities.append(np.eye(2))
    for i in range(N):
        HB += kron_list(identities[:i] + [pauli_X] + identities[i:])
    Ham += h * HB
    
    ## Get rid of small values
    return - Ham * (np.abs(Ham) > 1e-8).astype(np.int)
    
    
#Ham = create_hamiltonian(N=9, h=1.0)  
#en = np.linalg.eigvalsh(Ham)
#print(Ham.shape)
#print(en[:10])