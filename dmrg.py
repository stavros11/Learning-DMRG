# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:48:26 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf
from graphs import Operations
from numpy.linalg import svd

## Using the general eigh function because eigh_tridiagonal does not support complex! ##
## Change in the future for better performance ##
from scipy.linalg import eigh

def diagonalize(alpha, beta):
    ## Takes diagonal (alpha) and off-diagonal (beta) elements of
    ## tridiagonal matrix and returns its eigenvalues and eigenvectors
    d = len(alpha)
    ## Create matrix
    A = np.diag(alpha)
    A[np.arange(d-1), np.arange(1, d)] = beta
    A[np.arange(1,d), np.arange(d-1)] = beta
    
    return eigh(A)

###################################################################
### Only DMRG_Hlist is supported currently because tf.einsum    ###
### requires placeholders of specific shape                     ###
###################################################################

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
        print('\nStates succesfully initialized in right canonical form!\n')
        
        ## Create Ops object
        self.ops = Operations(D, H0, Hs, HN, lcz_k)
               
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
        self.state = [(np.random.random(size=(D1, D2, self.d)) + 1j * np.random.random(
                size=(D1, D2, self.d))).astype(np.complex64) for (D1, D2) in zip(self.D[:-1], self.D[1:])]
        ## First and last states are (d, d) (for boundary D=d in normal form)
        self.state = ([(np.random.random(size=(self.d, self.d)) + 1j * np.random.random(size=(self.d, self.d))).astype(np.complex64)] +
                       self.state + 
                       [(np.random.random(size=(self.d, self.d)) + 1j * np.random.random(size=(self.d, self.d))).astype(np.complex64)])
        
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
        self.R, self.L = (self.ops.N - 2) * [None], (self.ops.N - 2) * [None]
        self.update_R_boundary()
        for i in range(self.ops.N - 4, -1, -1):
            self.update_R(i)
            
    def sweep(self):
        ## Update left boundary
        energy_list = [self.apply_lanczos0()]
        self.update_L_boundary()
        
        ## Sweep to right
        energy_list.append(self.apply_lanczosM_to_right(1))
        self.update_L(1)
        for i in range(2, self.ops.N - 2):
            energy_list.append(self.apply_lanczosM_to_right(i))
            self.update_R(i-2)
            self.update_L(i)
            print('Site %d'%(i+1))
        
        print('\nRight sweep completed!\n')
        
        ## Update right boundary
        energy_list.append(self.apply_lanczosN())
        self.update_R_boundary()
        
        ## Sweep to left
        energy_list.append(self.apply_lanczosM_to_left(self.ops.N - 3))
        self.update_R(self.ops.N - 4)
        for i in range(self.ops.N - 4, 0, -1):
            energy_list.append(self.apply_lanczosM_to_left(i))
            self.update_R(i-1)
            self.update_L(i+1)
            print('Site %d'%(i+2))
        
        print('\nLeft sweep completed!\n')
        
        return energy_list
    
    #################################
    ##### Functions that assist #####
    #################################
    
    def apply_lanczos0(self):
        ## Apply Lanczos
        V_lz, alpha, beta = self.sess.run(self.ops.lanczos0, feed_dict={self.ops.plc.R[0] : self.R[0],
                                                                        self.ops.plc.state[0] : self.state[0],
                                                                        self.ops.plc.state[1] : self.state[1]})
        #V: Lanczos vectors (see lanczos_algorithm functions in lanczos.py)
        # alpha: diagonal elements of the tridiagonal matrix, beta: off-diagonal elements
        
        ## Diagonalize k x k matrix
        eig_vals, eig_vec = diagonalize(alpha, beta)
        ## Transform the ground state eigenvector to B
        B = np.einsum('a,abcd->bcd', eig_vec[0], V_lz)
        
        ## Update states by doing SVD on the updated B
        self.energy = eig_vals[0]
        self.state[0], S, V = svd(B.reshape(self.d, self.D[1]*self.d), full_matrices=False)
        self.state[1] = np.einsum('ab,bcd->acd', np.diag(S), V.reshape(self.d, self.D[1], self.d))
        
        return self.energy
        
    def apply_lanczosN(self):
        V_lz, alpha, beta = self.sess.run(self.ops.lanczosN, feed_dict={self.ops.plc.L[-1] : self.L[-1],
                                                                        self.ops.plc.state[-1] : self.state[-2],
                                                                        self.ops.plc.state[0] : self.state[-1]})
        
        ## Diagonalize k x k matrix
        eig_vals, eig_vec = diagonalize(alpha, beta)
        ## Transform the ground state eigenvector to B
        B = np.einsum('a,abcd->bcd', eig_vec[0], V_lz)
        
        ## Updates
        self.energy = eig_vals[0]
        U, S, self.state[-1] = svd(B.reshape(self.D[-2]*self.d, self.d), full_matrices=False)
        self.state[-2] = np.einsum('abc,cd->adb', U.reshape(self.D[-2], self.d, self.d), np.diag(S))
        
        return self.energy
        
    def apply_lanczos_for_B(self, i):
        V_lz, alpha, beta = self.sess.run(self.ops.lanczosM[i-1], feed_dict={self.ops.plc.L[i-1] : self.L[i-1],
                                          self.ops.plc.R[i] : self.R[i],
                                          self.ops.plc.state[i] : self.state[i],
                                          self.ops.plc.state[i+1] : self.state[i+1]})
        
        ## Diagonalize k x k matrix
        eig_vals, eig_vec = diagonalize(alpha, beta)
        ## Transform the ground state eigenvector to B
        B = np.einsum('a,abcde->bcde', eig_vec[0], V_lz)
        
        return B, eig_vals[0]
        
    def apply_lanczosM_to_right(self, i):
        ## Here i is the index of the state to be updated: Hence 1 <= i <= N-3
        ## For i=0 use lanczos0, for i=N-1 use lanczosN
        B, self.energy = self.apply_lanczos_for_B(i)
        U, S, V = svd(B.reshape(self.D[i-1]*self.d, self.D[i+1]*self.d))
        ## Assume Di < d D_{i-1} and truncate
        U, S, V = U[:, :self.D[i]], S[:self.D[i]], V[:self.D[i]]
        
        ## Updates
        self.state[i] = np.transpose(U.reshape(self.D[i-1], self.d, self.D[i]), axes=(0, 2, 1))
        self.state[i+1] = np.transpose((np.diag(S).dot(V)).reshape(self.D[i], self.d, self.D[i+1]), 
                  axes=(0, 2, 1))
        
        return self.energy
    
    def apply_lanczosM_to_left(self, i):
        ## Here i+1 is the index of the state to be updated. 1 <= i <= N-3
        B, self.energy = self.apply_lanczos_for_B(i)
        U, S, V = svd(B.reshape(self.D[i-1]*self.d, self.D[i+1]*self.d))
        ## Assume Di < d D_{i-1} and truncate
        U, S, V = U[:,:self.D[i]], S[:self.D[i]], V[:self.D[i]]
        
        ## Updates
        self.state[i+1] = np.transpose(V.reshape(self.D[i], self.d, self.D[i+1]), axes=(0, 2, 1))
        self.state[i] = np.transpose((U.dot(np.diag(S))).reshape(self.D[i-1], self.d, self.D[i]), 
                  axes=(0, 2, 1))
        
        return self.energy
        
    def update_L(self, i):
        ## Here i is the index of L to be updated: 1 <= i <= N-3
        ## For i=0 use boundary function
        self.L[i] = self.sess.run(self.ops.L[i], feed_dict={self.ops.plc.L[i-1] : self.L[i-1], 
              self.ops.plc.state[i] : self.state[i]})
    
    def update_R(self, i):
        ## Here i is the index of L to be updated: 0 <= i <= N-4
        ## For i=N-3 use boundary function
        self.R[i] = self.sess.run(self.ops.R[i], feed_dict={self.ops.plc.R[i+1] : self.R[i+1],
              self.ops.plc.state[i+2] : self.state[i+2]})
    
    def update_R_boundary(self):
        self.R[-1] = self.sess.run(self.ops.R[-1], feed_dict={self.ops.plc.state[0] : self.state[-1]})
        
    def update_L_boundary(self):
        self.L[0] = self.sess.run(self.ops.L[0], feed_dict={self.ops.plc.state[0] : self.state[0]})
