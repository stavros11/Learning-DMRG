# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:31:36 2018

@author: Admin
"""

import numpy as np
from itertools import product
from os import path
from TFIM1D_ED import create_hamiltonian

from keras.models import Sequential
from keras.layers import Conv1D, InputLayer, Lambda
import keras.backend as K

#################################################
## Brute force approach by creating all states ##
#################################################

def classical_energy(states, pbc=False):
    x = 1 - 2 * states
    energy = np.sum(x[:, 1:] * x[:, :-1], axis=1)
    if pbc:
        energy+= x[:, 0] * x[:, 1]
    return -energy

class Predictor(object):
    def __init__(self, N, h, pbc=False):
        ## Number of sites
        self.N = N
        ## Magnetic field
        self.h = h
        ## Periodic Boundary Conditions
        self.pbc = pbc
        
        ## Create all possible states
        self.states = np.array([list(x) for x in product([0, 1], repeat=self.N)])
        
        ## Create classical quantity required for energy calculation
        self.classical = classical_energy(self.states, pbc=self.pbc)
        self.to_decimal = 2**np.arange(self.N-1, -1, -1)
        
    def compiler(self, folder=None):       
        self.machine()
        self.load_model(folder)
            
    def machine(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.states.shape[1:]+(1,)))
        self.model.add(Conv1D(32, 4, activation='relu'))
        self.model.add(Conv1D(16, 3, activation='relu'))
        self.model.add(Conv1D(8, 3, activation='sigmoid'))
        #dim = K.get_variable_shape(self.model.layers[-1].output)[1]
        #self.model.add(AveragePooling1D(pool_size=dim))
        #self.model.add(GlobalAveragePooling1D())
        self.model.add(Lambda(lambda x : K.mean(x, axis=(1,2))))
        
    def predict(self):
        return self.model.predict(self.states[:, :, np.newaxis])
                
#    def energy(self, psi):
#        psi2 = psi * psi
#        self.Z = K.sum(psi2)
#        
#        interaction = K.sum(psi2 * self.classical)
#        field = self.field_term(0, psi)
#        for i in range(1, self.N):
#            field = field + self.field_term(i, psi)
#
#        return (interaction - self.h * field) / self.Z
#        
#    def field_term(self, i, psi):
#        flipped_states = np.copy(self.states)
#        flipped_states[:,i] = (self.states[:,i] == 0).astype(np.int)
#        ind = flipped_states.dot(self.to_decimal)
#        
#        return K.sum(psi * psi[ind])
#    
#    def loss(self, y_true, y_pred):
#        return self.energy(y_pred[:, 0, 0])
#    
#    def save_weights(self, folder):
#        for (i, l) in enumerate(self.model.layers):
#            np.save(path.join(folder, 'WeightsLayer%d.npy'%i), K.eval(l.weights[0]))
#            np.save(path.join(folder, 'BiasesLayer%d.npy'%i), K.eval(l.weights[1]))
            
    def load_model(self, folder):
        for i in range(len(self.model.layers[:-1])):
            self.model.layers[i].set_weights([np.load(path.join(folder, 'WeightsLayer%d.npy'%i)),
                                              np.load(path.join(folder, 'BiasesLayer%d.npy'%i))])
        print('\nWeights loaded.\n')
        

N = 9
tr = Predictor(N=N, h=1.0)
tr.compiler('CNN_N8_pool')

psi = tr.predict()

Ham = create_hamiltonian(N=N, h=1.0)
print(np.sum(psi * (Ham.dot(psi))) / np.sum(psi * psi))