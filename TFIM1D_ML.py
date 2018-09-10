# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:03:20 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf
from itertools import product
from os import path

from keras.models import Sequential
from keras.layers import Conv1D, InputLayer
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
        
    def compiler(self, folder=None):
        ## Create wavefunction and placeholders
        self.plc = tf.placeholder(shape=self.states.shape+(1,), dtype=tf.float32)
        #self.psi = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=(2**N,)), dtype=tf.float32)
        
        self.machine()
        if folder != None:
            self.load_model(folder)
        
        self.psi = self.model.layers[-1].output[:,0,0]
    
        ## Create energy and training ops
        self.energy_graph()
        self.training_graph()
        
    def machine(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_tensor=self.plc))
        self.model.add(Conv1D(64, 5, activation='relu'))
        self.model.add(Conv1D(32, 4, activation='relu'))
        self.model.add(Conv1D(1, 3, activation='sigmoid'))
                
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
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.energy_op)
    
    def train_early_stop(self, folder, delta=1e-8, patience=20, en_calc=100, message=2000):
        fd = {self.plc : self.states[:,:,np.newaxis]}
        epoch, counter = 0, 0
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            energies, psis = [sess.run(self.energy_op, feed_dict=fd)], [sess.run(self.psi, feed_dict=fd)]
            
            while True:
                sess.run(self.train_op, feed_dict=fd)
                
                if epoch % en_calc == 0:
                    psis.append(sess.run(self.psi, feed_dict=fd))
                    energies.append(sess.run(self.energy_op, feed_dict=fd))
                    
                    if epoch % message == 0:
                        print('Epoch: %d  -  Energy: %.10f\n'%(epoch, energies[-1]))
                        
                if np.abs(energies[-1] - energies[-2]) < delta:
                    counter += 1
                else:
                    counter = 0
                    
                if counter > patience:
                    break
                
                epoch += 1
                
            self.save_weights(sess, folder)
        
        return energies, psis
    
    def save_weights(self, sess, folder):
        for (i, l) in enumerate(self.model.layers):
            np.save(path.join(folder, 'WeightsLayer%d.npy'%i), sess.run(l.weights[0]))
            np.save(path.join(folder, 'BiasesLayer%d.npy'%i), sess.run(l.weights[1]))
            
    def load_model(self, folder):
        for i in range(len(self.model.layers)):
            self.model.layers[i].set_weights([np.load(path.join(folder, 'WeightsLayer%d.npy'%i)),
                                              np.load(path.join(folder, 'BiasesLayer%d.npy'%i))])
        print('\nWeights loaded.\n')
    
### Debugging ###
tr = Trainer(N=10, h=1.0)
tr.compiler()

en_pred, psis = tr.train_early_stop('CNN1', patience=5, en_calc=1, message=100)

#tr.compiler('CNN2')

