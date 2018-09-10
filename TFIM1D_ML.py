# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:03:20 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf
from itertools import product

#################################################
## Brute force approach by creating all states ##
#################################################

def classical_energy(states, pbc=False):
    x = 1 - 2 * states
    energy = np.sum(x[:, 1:] * x[:, :-1], axis=1)
    if pbc:
        energy+= x[:, 0] * x[:, 1]
    return -energy

class Trainer(object):
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
        self.classical = tf.constant(classical_energy(self.states, pbc=self.pbc), dtype=tf.float32)
        self.to_decimal = 2**np.arange(self.N-1, -1, -1)
        
        ## Create wavefunction
        #self.psi = tf.Variable(np.load('TFIM1D_N10H1_GS.npy'), dtype=tf.float32)
        self.psi = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=(2**N,)), dtype=tf.float32)
    
        ## Create energy and training ops
        self.energy_graph()
        self.training_graph()
                
    def energy_graph(self):
        psi2 = tf.multiply(self.psi, self.psi)
        self.Z = tf.reduce_sum(psi2)
        
        interaction = tf.reduce_sum(tf.multiply(psi2, self.classical))
        field = self.field_term(0)
        for i in range(1, self.N):
            field = tf.add(field, self.field_term(i))

        self.energy_op = (interaction - self.h * field) / self.Z
    
    def field_term(self, i):
        flipped_states = np.copy(self.states)
        flipped_states[:,i] = (self.states[:,i] == 0).astype(np.int)
        ind = flipped_states.dot(self.to_decimal)
        
        return tf.reduce_sum(tf.multiply(self.psi,
                                         tf.gather(self.psi, ind)))
    
    def training_graph(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.energy_op)
    
    def train_early_stop(self, delta=1e-8, patience=20, en_calc=100, message=2000):
        #fd = {self.machine.states : self.states}
        epoch, counter = 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            energies, psis = [sess.run(self.energy_op)], [sess.run(self.psi)]
            
            while True:
                sess.run(self.train_op)
                
                if epoch % en_calc == 0:
                    psis.append(sess.run(self.psi))
                    energies.append(sess.run(self.energy_op))
                    
                    if epoch % message == 0:
                        print('Epoch: %d  -  Energy: %.10f\n'%(epoch, energies[-1]))
                        
                if np.abs(energies[-1] - energies[-2]) < delta:
                    counter += 1
                else:
                    counter = 0
                    
                if counter > patience:
                    break
                
                epoch += 1
        
        return energies, psis
    
### Debugging ###
tr = Trainer(N=10, h=1.0)

en_pred, psis = tr.train_early_stop(patience=10, en_calc=1, message=50)

#### Calculate normal quantities ###
#from TFIM1D_ED import Ham
#cl = classical_energy(tr.states) - tr.h * classical_magnetization(tr.states) / np.sqrt(2)
#
#t1_th, t2_th, Z_th = np.zeros(len(psis)), np.zeros(len(psis)), np.zeros(len(psis))
#en_th, en_thc = np.zeros(len(psis)), np.zeros(len(psis))
#for (i, x) in enumerate(psis):
#    Z_th[i] = np.sum(x * x)
#    en_th[i] = np.sum((x * Ham.dot(x)) / Z_th[i])
#    t1_th[i] = np.sum(x*x*cl)
#    for j in range(tr.N):
#        t2_th[i] += np.sum(x[tr.selection[j]] * x[tr.selection[j] + 2**(9-j)])
#    
#    en_thc[i] = (t1_th[i] - np.sqrt(2.0) * tr.h * t2_th[i]) / Z_th[i]