# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:48:26 2018

@author: Admin
"""

import numpy as np
import tensorflow as tf

class NumPy_Tensors():
    def __init__(self, D, H0, Hs, HN):
        ### DMRG parameters ###
        ## N: Number of sites in the chain
        N = len(Hs) + 2
        ## d: Physical dimension of each degree of freedom
        d = len(H0)
        ## D: MPS bond dimension (same for all sites)
        
        ### Hamiltonian in MPO form ###
        ## DH: Hidden dimension for Hamiltonian
        ## H0: Hamiltonian for first site: Complex (DH, d, d)
        ## Hs: Hamiltonian for middle chain: Complex (N-2, DH, DH, d, d)
        ## HN: Hamiltonian for last site: Complex (DH, d, d)
        
        # Initialize MPS: Complex (N-2, D, D, d)
        # First and last states are (D, d)
        self.state0 = np.random.random(size=(d, D)) + 1j * np.random.random(size=(D, d))
        self.states = np.random.random(size=(N-2, D, D, d)) + 1j * np.random.random(size=(N-2, D, D, d))
        self.stateN = np.random.random(size=(d, D)) + 1j * np.random.random(size=(D, d))
        
        # Initialize Hamiltonian in MPO form
        self.H0, self.Hs, self.HN = H0, Hs, HN
    

class Operations():
    def __init__(self, D, H0, Hs, HN):
        self.N = len(Hs) + 2
        DH, d = H0.shape[:2]
        
        # Initialize Hamiltonian for TF graph
        self.H0 = tf.constant(H0, dtype=tf.complex64)
        self.Hs = tf.constant(Hs, dtype=tf.complex64)
        self.HN = tf.constant(HN, dtype=tf.complex64)
        
        # Create placeholders for MPS and R tensors
        self.state_boundary = tf.placeholder(dtype=tf.complex64, shape=(D, d))
        self.state = tf.placeholder(dtype=tf.complex64, shape=(D, D, d))
        self.R = tf.placeholder(dtype=tf.complex64, shape=(D, DH, D))
        self.L = tf.placeholder(dtype=tf.complex64, shape=(D, DH, D))
        
        self.calc_R, self.calc_L = self.RL_boundary_graph(self.HN), self.RL_boundary_graph(self.H0)
        ## Rs[i][0] for right, Rs[i][1] for left ##
        self.calc_RLs = [[self.R_graph(i), self.L_graph(i)] for i in range(self.N-2)]
        
    def initialize_MPS(self, state0, states, stateN):
        # Create variables for optimization
        ## (fix initializations here) ##       
        self.B0 = tf.Variable(np.einsum('ai,abj->bij', self.state0, self.states[0]), dtype=tf.complex64)
        self.BN = tf.Variable(np.einsum('bj,abi->aij', self.stateN, self.states[-1]), dtype=tf.complex64)
        self.Bs = [tf.Variable(np.einsum('abi,bcj->acij', self.state[i], self.state[i+1]), 
                               dtype=tf.complex64) for i in range(self.N-2)]
        
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
    
    #### CAN DELETE ####
    def HB0_graph(self, B):
        RB = tf.einsum('abc,cij->abij', self.R, B)
        RB = tf.einsum('abij,dbkj->adik', RB, self.Hs[0])
        return tf.einsum('adik,dli->alk', RB, self.H0)
    
    def HBN_graph(self, B):
        LB = tf.einsum('abc,cij->abij', self.L, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.Hs[-1])
        return tf.einsum('adkj,dlj->akl', LB, self.HN)
    
    def HB_middle_graph(self, B, i):
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.Hs[i])
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.Hs[i+1])
        return tf.einsum('afgij,efg->aeij', LB, self.R)
    

    