# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:48:26 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf
from graphs import Operations
from scipy.linalg import eigh_tridiagonal as eigtrd
from numpy.linalg import svd

class DMRG(object):
    def __init__(self, D, d, H0, Hs, HN, lcz_k):
        ### DMRG parameters ###
        ## d: Physical dimension of each degree of freedom
        ## D: List with MPS matrices dimensions (the list ignores first and last site where d=D)
        ## lcz_k: k-number for Lanczos
        self.d, self.D = d, D
        
        ## Initialize and normalize states to canonical form
        self.initialize_states()
        self.normalize_states()
        print('\nStates succesfully initialized in right canonical form!')
        
        ## Create Ops object
        self.ops = Operations(D, H0, Hs, HN, lcz_k, self.plc)
               
        ## Open tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        ## Initialize R, L matrices and calculate R
        self.initialize_RL()
        print('R succesfully initialized!')
    
    #############################################
    #### Functions that run basic operations ####
    #############################################
    
    def initialize_states(self):
        ## Initialize MPS: List of complex (D, D, d) tensors
        self.state = [np.random.random(size=(D1, D2, self.d)) + 1j * np.random.random(
                size=(D1, D2, self.d)) for (D1, D2) in zip(self.D[:-1], self.D[1:])]
        ## First and last states are (d, d) (for boundary D=d in normal form)
        self.state = ([np.random.random(size=(self.d, self.d)) + 1j * np.random.random(size=(self.d, self.d))] +
                       self.state + [np.random.random(size=(self.d, self.d)) + 1j * np.random.random(size=(self.d, self.d))])
        
    def normalize_states(self):
        ## Start from right
        U, S, self.state[-1] = svd(self.state[-1], full_matrices=False)
        # (d x d)(d x d stored as (d,))(d x d)
        self.state[-2] = np.einsum('abi,bc->aci', self.state[-2], U.dot(np.diag(S)))
        # (D, D', d) = (D, D', d) x (D', D') (contraction of second index)
        
        ## Repeat for middle states
        for i in range(len(self.D) - 1, 1, -1):
            U, S, V = svd(self.state[i].reshape(self.D[i-1], self.D[i]*self.d), full_matrices=False)
            self.state[i] = V.reshape(self.D[i-1], self.D[i], self.d)
            self.state[i-1] = np.einsum('abi,bc->aci', self.state[i-1], U.dot(np.diag(S)))
        
        ## Normalize final state
        U, S, V = svd(self.state[1].reshape(self.D[0], self.D[1]*self.d), full_matrices=False)
        self.state[1] = V.reshape(self.D[0], self.D[1], self.d)
        self.state[0] = np.einsum('bi,bc->ci', self.state[0], U.dot(np.diag(S)))
        U, S, self.state[0] = svd(self.state[0], full_matrices=False)
        
    def initialize_RL(self):       
        self.R, self.L = (self.ops.N - 1) * [None], (self.ops.N - 1) * [None]
        self.update_R_boundary()
        for i in range(self.ops.N - 3, -1, -1):
            self.update_R(i)
            
    def sweep(self):
        ## Update left boundary
        energy_list = [self.apply_lanczos0()]
        self.update_L_boundary()
        
        ## Sweep to right
        for i in range(1, self.ops.N - 1):
            energy_list.append(self.apply_lanczosM(i))
            self.update_R(i-1)
            self.update_L(i)
        
        ## Update right boundary
        energy_list.append(self.apply_lanczosN())
        self.update_R_boundary()
        
        ## Sweep to left
        for i in range(self.ops.N - 2, 0, -1):
            energy_list.append(self.apply_lanczosM(i))
            self.update_R(i-1)
            self.update_L(i)
        
        return energy_list
    
    ##### IMPORTANT! #######
    ### Also check dimensions in lanczos ops in graphs.py module!
    
    #################################
    ##### Functions that assist #####
    #################################
    
    def apply_lanczos0(self):
        ## Apply Lanczos
        U, V, alpha, beta = self.sess.run(self.ops.lanczos0, feed_dict={self.ops.plc.R : self.R[0]})
        # U: probably useless, V: Lanczos right vectors stored as rows!
        # alpha: diagonal elements of the bidiagonal matrix, beta[:-1]: off-diagonal elements
        eig_vals, eig_vec = eigtrd(alpha, beta[:-1])
        
        ## Update states by doing SVD on the updated B = V.dot(eigenvector)
        self.energy = eig_vals[0]
        self.state[0], S, V = svd((V.dot(eig_vec[:, 0])).reshape(self.d, self.D[1]*self.d), full_matrices=False)
        self.state[1] = np.einsum('ab,bcd->acd', np.diag(S), V.reshape(self.d, self.D[1], self.d))
        
        return self.energy
        
    def apply_lanczosN(self):
        U, V, alpha, beta = self.sess.run(self.ops.lanczosN, feed_dict={self.ops.plc.L : self.L[-1]})
        eig_vals, eig_vec = eigtrd(alpha, beta[:-1])
        
        ## Updates
        self.energy = eig_vals[0]
        U, S, self.state[-1] = svd((V.dot(eig_vec[:, 0])).reshape(self.D[-2]*self.d, self.d), full_matrices=False)
        self.state[-2] = np.einsum('abc,cd->adb', U.reshape(self.D[-2], self.d, self.d), np.diag(S))
        
        return self.energy
        
    def apply_lanczosM(self, i):
        ## Here i is the index of the state to be updated: Hence 1 <= i <= N-2
        ## For i=0 use lanczos0, for i=N-1 use lanczosN
        U, V, alpha, beta = self.sess.run(self.ops.lanczosM[i-1], feed_dict={self.ops.plc.L : self.L[i-1],
                                          self.ops.plc.R : self.R[i+1]})
        eig_vals, eig_vec = eigtrd(alpha, beta[:-1])
        
        U, S, V = svd((V.dot(eig_vec[:, 0])).reshape(self.D[i-1]*self.d, self.D[i+1]*self.d))
        ## Assume Di < d D_{i-1} and truncate
        U, S, V = U[:, :self.D[i]], S[:self.D[i]], V[:self.D[i]]
        
        ## Updates
        self.energy = eig_vals[0]
        self.state[i] = U.reshape(self.D[i-1], self.d, self.D[i]).transpose(axes=(0, 2, 1))
        self.state[i+1] = (np.diag(S).dot(V)).reshape(self.D[i], self.d, self.D[i+1]).transpose(axes=(0, 2, 1))
        
        return self.energy
        
    def update_L(self, i):
        ## Here i is the index of L to be updated: 1 <= i <= N-1
        ## For i=0 use boundary function
        self.L[i] = self.sess.run(self.ops.L, feed_dict={self.ops.plc.L : self.L[i-1], 
              self.ops.plc.state : self.state[i]})
    
    def update_R(self, i):
        ## Here i is the index of L to be updated: 0 <= i <= N-3
        ## For i=N-2 use boundary function
        self.R[i] = self.sess.run(self.ops.R, feed_dict={self.ops.plc.R : self.R[i+1],
              self.ops.plc.state : self.state[i+1]})
    
    def update_R_boundary(self):
        self.R[-1] = self.sess.run(self.ops.R_boundary, feed_dict={self.ops.plc.state : self.state[-1]})
        
    def update_L_boundary(self):
        self.L[0] = self.sess.run(self.ops.L_boundary, feed_dict={self.ops.plc.state : self.state[0]})
        

################################################
#### For the case where Hs is given as list ####
################################################
class DMRG_Hlist(DMRG):
    def initialize_RL(self):
        ## Calculate first R (begining from right)
        self.R = [self.sess.run(self.ops.R_boundary, feed_dict={self.ops.plc.state : self.state[-1]})]       
        for i in range(self.ops.N - 2):
            self.R.append(self.sess.run(self.ops.R[self.ops.N - 3 - i], feed_dict={self.ops.plc.R : self.R[i], 
                                        self.ops.plc.state : self.states[self.ops.N - 2 - i]}))
    
        self.R = self.R[::-1]
        self.L = (self.ops.N - 1) * [None]
        
    def update_L(self, i):
        ## Here i is the index of L to be updated: 1 <= i <= N-2
        ## For i=0 use boundary function
        self.L[i] = self.sess.run(self.ops.L[i-1], feed_dict={self.ops.plc.L : self.L[i-1], 
              self.ops.plc.state : self.state[i]})
    
    def update_R(self, i):
        ## Here i is the index of L to be updated: 0 <= i <= N-3
        ## For i=N-2 use boundary function
        self.R[i] = self.sess.run(self.ops.R[i], feed_dict={self.ops.plc.R : self.R[i+1],
              self.ops.plc.state : self.state[i]})