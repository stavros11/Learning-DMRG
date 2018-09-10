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

def classical_magnetization(states):
    #return np.sum(2 * states - 1, axis=1)
    return np.sum(states, axis=1)

def classical_energy(states, pbc=False):
    #x = 2 * states - 1
    x = states
    energy = np.sum(x[:, 1:] * x[:, :-1], axis=1)
    if pbc:
        energy+= x[:, 0] * x[:, 1]
    return -energy

class Machine(object):
    def __init__(self, N):
        self.N = N
        
        #self.psi = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size=(2**N,)), 
        #                       dtype=tf.float32)
        self.psi = tf.Variable(np.load('TFIM1D_N10H1_GS.npy'), dtype=tf.float32)

class Trainer(object):
    def __init__(self, machine, h, pbc=False):
        ## Number of sites
        self.N = machine.N
        ## Machine that gives psi(sigma)
        self.machine = machine
        ## Magnetic field
        self.h = h
        ## Periodic Boundary Conditions
        self.pbc = pbc
        
        ## Create all possible states
        self.states = np.array([list(x) for x in product([1, -1], repeat=self.N)])
        
        ## Create classical quantity required for energy calculation
        self.classical = tf.constant((classical_energy(self.states, pbc=self.pbc)
        - self.h * classical_magnetization(self.states)), dtype=tf.float32)
    
        ## Selection indices for cross term
        self.selection = np.array([np.argwhere(self.states[:,i] == -1) for i in range(self.N)])[:,:,0]
    
        ## Create energy and training ops
        self.energy_op = self.energy_graph()
        self.train_op = self.training_graph()
        
    def energy_graph(self):
        psi2 = tf.multiply(self.machine.psi, self.machine.psi)
        print(psi2)
        self.Z = tf.reduce_sum(psi2)
        
        classical_term = tf.reduce_sum(tf.multiply(psi2, self.classical))
        print(classical_term)
        cross_term = tf.reduce_sum(tf.multiply(
                    tf.gather(self.machine.psi, self.selection[0]),
                    tf.gather(self.machine.psi, self.selection[0] - 2**(9))))
        for i in range(1, self.N):
            cross_term = tf.add(cross_term, tf.reduce_sum(tf.multiply(
                    tf.gather(self.machine.psi, self.selection[i]),
                    tf.gather(self.machine.psi, self.selection[i] - 2**(9-i)))))
        
        print(cross_term)
        return (classical_term - 2 * self.h * cross_term) / self.Z
    
    def training_graph(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        return optimizer.minimize(self.energy_op)
    
    def train_early_stop(self, delta=1e-8, patience=20, en_calc=100, message=2000):
        #fd = {self.machine.states : self.states}
        epoch, counter = 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            energies, psis = [sess.run(self.energy_op)], [sess.run(self.machine.psi)]
            
            while True:
                sess.run(self.train_op)
                
                if epoch % en_calc == 0:
                    energies.append(sess.run(self.energy_op))
                    psis.append(sess.run(self.machine.psi))
                    
                    if epoch % message == 0:
                        print('Epoch: %d  -  Energy: %.10f\n'%(epoch, energies[-1]))
                        
                if np.abs(energies[-1] - energies[-2]) < delta:
                    counter += 1
                    
                if counter > patience:
                    break
                
                epoch += 1
        
        return energies, psis