# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:48:26 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf
import lanczos as lcz
from scipy.linalg import eigh_tridiagonal as eigtrd
from numpy.linalg import svd

class NumPy_Tensors(object):
    def __init__(self, D, H0, Hs, HN, lcz_k):
        ### DMRG parameters ###
        ## N: Number of sites in the chain
        N = len(Hs) + 2
        ## d: Physical dimension of each degree of freedom
        self.d = len(H0)
        ## D: MPS bond dimension (same for all sites)
        self.D = D
        ## lcz_k: k-number for Lanczos
        
        ### Hamiltonian in MPO form ###
        ## DH: Hidden dimension for Hamiltonian
        ## H0: Hamiltonian for first site: Complex (DH, d, d)
        ## Hs: Hamiltonian for middle chain: Complex (N-2, DH, DH, d, d)
        ## HN: Hamiltonian for last site: Complex (DH, d, d)
        
        # Initialize MPS: Complex (N-2, D, D, d)
        # First and last states are (D, d)
        self.state0 = np.random.random(size=(self.d, D)) + 1j * np.random.random(size=(D, self.d))
        self.states = np.random.random(size=(N-2, D, D, self.d)) + 1j * np.random.random(size=(N-2, D, D, self.d))
        self.stateN = np.random.random(size=(self.d, D)) + 1j * np.random.random(size=(D, self.d))
        
        ## Create Ops object
        self.ops = Operations(D, H0, Hs, HN, lcz_k)
        
        ## Open session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def first_update(self):
        ## Keep Rs in memory
        self.R = np.zeros((self.ops.N - 1, self.D, self.ops.DH, self.D))
        
        ## Calculate Rs
        self.R[-1] = self.sess.run(self.ops.calc_R, feed_dict={self.ops.state_boundary : self.stateN})
        for i in range(self.ops.N - 3, -1 , -1):
            self.R[i] = self.sess.run(self.ops.calc_RLs[i][0], feed_dict={self.ops.R : self.R[i+1],
                  self.ops.state : self.states[i]})
    
        ## Apply Lanczos
        U, V, alpha, beta = self.sess.run(self.ops.lanczos_boundary0, feed_dict={self.ops.R : self.R[0]})
        # U: probably useless, V: Lanczos right vectors stored as rows!
        # alpha: diagonal elements of the bidiagonal matrix, beta[:-1]: off-diagonal elements
        eig_vals, eig_vec = eigtrd(alpha, beta[:-1])
        ## Update B and do SVD
        U, S, V = svd((V.dot(eig_vec[:, 0])).reshape(self.d, self.D*self.d))
        
        ## Update states
        state0 = 
    

class Operations(object):
    def __init__(self, D, H0, Hs, HN, lcz_k):
        self.N = len(Hs) + 2
        self.DH, d = H0.shape[:2]
        
        ## Initialize Hamiltonian for TF graph
        self.H0 = tf.constant(H0, dtype=tf.complex64)
        self.Hs = tf.constant(Hs, dtype=tf.complex64)
        self.HN = tf.constant(HN, dtype=tf.complex64)
        
        ## Create placeholders for MPS and R tensors
        self.state_boundary = tf.placeholder(dtype=tf.complex64, shape=(D, d))
        self.state = tf.placeholder(dtype=tf.complex64, shape=(D, D, d))
        self.R = tf.placeholder(dtype=tf.complex64, shape=(D, self.DH, D))
        self.L = tf.placeholder(dtype=tf.complex64, shape=(D, self.DH, D))
        
        ## Create ops for R and L calculations/updates
        self.calc_R, self.calc_L = self.RL_boundary_graph(self.HN), self.RL_boundary_graph(self.H0)
        ## Rs[i][0] for right, Rs[i][1] for left ##
        self.calc_RLs = [[self.R_graph(i), self.L_graph(i)] for i in range(self.N-2)]
        
        ## Create Lanczos ops
        self.lanczos_boundary0 = tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops_Boundary(D, d, self.H0, self.Hs[0], self.R), k=lcz_k, name="lanczos_bidiag_boundary0")
        self.lanczos_boundaryN = tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops_Boundary(D, d, self.Hs[-1], self.HN, self.L), k=lcz_k, name="lanczos_bidiag_boundaryN")
        self.lanczos = [tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops(D, d, self.Hs[i], self.Hs[i+1], self.L, self.R), k=lcz_k) for i in range(self.N-3)]
        
    
    ##############################
    ###### Is this needed ? ######
    def initialize_MPS(self, state0, states, stateN):
        # Create variables for optimization
        ## (fix initializations here) ##       
        self.B0 = tf.Variable(np.einsum('ai,abj->bij', self.state0, self.states[0]), dtype=tf.complex64)
        self.BN = tf.Variable(np.einsum('bj,abi->aij', self.stateN, self.states[-1]), dtype=tf.complex64)
        self.Bs = [tf.Variable(np.einsum('abi,bcj->acij', self.state[i], self.state[i+1]), 
                               dtype=tf.complex64) for i in range(self.N-2)]
    #############################
        
    def RL_boundary_graph(self, hamiltonian):
        R = tf.einsum('bij,cj->bci', hamiltonian, self.state_boundary)
        return tf.einsum('bci,ai->abc', R, tf.conj(self.state_boundary))
    
    def R_graph(self, i):
        Rnew = tf.einsum('abc,fcj->abfj', self.R, self.state)
        Rnew = tf.einsum('ebij,abfj->aefi', self.Hs[i], Rnew)
        return tf.einsum('adi,aefi->def', tf.conj(self.state), Rnew)
        #also changed the indices in einsum to take into account the dagger
    
    def L_graph(self, i):
        Lnew = tf.einsum('def,fcj->decj', self.L, self.state)
        Lnew = tf.einsum('ebij,decj->dbci', self.Hs[i], Lnew)
        return tf.einsum('adi,dbci->abc', tf.conj(self.state), Lnew)
        #also changed the indices in einsum to take into account the dagger

    