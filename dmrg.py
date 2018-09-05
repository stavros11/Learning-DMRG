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

class DMRG(object):
    def __init__(self, D, d, H0, Hs, HN, lcz_k):
        ### DMRG parameters ###
        ## d: Physical dimension of each degree of freedom
        ## D: List with MPS matrices dimensions (the list ignores first and last site where d=D)
        ## lcz_k: k-number for Lanczos
        self.d, self.D = d, D
        
        ### Hamiltonian in MPO form ###
        ## DH: Hidden dimension for Hamiltonian
        ## H0: Hamiltonian for first site: Complex (DH, d, d)
        ## Hs: Hamiltonian for middle chain: Complex (N-2, DH, DH, d, d)
        ## HN: Hamiltonian for last site: Complex (DH, d, d)
        
        ## Initialize MPS: List of complex (D, D, d) tensors
        self.state = [np.random.random(size=(D1, D2, d)) + 1j * np.random.random(
                size=(D1, D2, d)) for (D1, D2) in zip(D[:-1], D[1:])]
        ## First and last states are (d, d) (for boundary D=d in normal form)
        self.state = ([np.random.random(size=(d, d)) + 1j * np.random.random(size=(d, d))] +
                       self.state + [np.random.random(size=(d, d)) + 1j * np.random.random(size=(d, d))])
        
        ## Normalize states to canonical form
        self.normalize_states()
        print('\nStates succesfully normalized!')
        
        ## Create Ops object
        self.ops = Operations(D, H0, Hs, HN, lcz_k, self.plc)
        
        ## Open session
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        
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
        
    def initialize_R(self):
        ## Keep Rs in memory
        self.R = []
        
        ## Calculate first R (begining from right)
        self.R.append(self.sess.run(self.ops.calc_R, feed_dict={self.ops.plc.state : self.state[-1]}))
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
        
class Placeholders(object):
    def __init__(self):
        self.state = tf.placeholder(dtype=tf.complex64)
        self.R = tf.placeholder(dtype=tf.complex64)
        self.L = tf.placeholder(dtype=tf.complex64)
        
class Hamiltonian(object):
    def __init__(self, H0, Hs, HN):
        self.left = tf.constant(H0, dtype=tf.complex64)
        self.mid = tf.constant(Hs, dtype=tf.complex64)
        self.right = tf.constant(HN, dtype=tf.complex64)
    
class Operations(object):
    def __init__(self, D, H0, Hs, HN, lcz_k):
        self.N = len(Hs) + 2
        self.DH, d = H0.shape[:2]
        
        ## Create placeholders object
        self.plc = Placeholders()
        ## Create hamiltonian object
        self.H = Hamiltonian(H0, Hs, HN)
        
        ## Determine if Hamiltonian is list or same for all middle sites
        self.H_list_flag = (len(Hs.shape) >= 5)
        
        ## Create Lanczos ops
        self.lanczos_boundary0 = tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops_Boundary(D, d, self.H0, self.Hs[0], self.R), k=lcz_k, name="lanczos_bidiag_boundary0")
        self.lanczos_boundaryN = tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops_Boundary(D, d, self.Hs[-1], self.HN, self.L), k=lcz_k, name="lanczos_bidiag_boundaryN")
        self.lanczos = [tf.contrib.solvers.lanczos.lanczos_bidiag(
                operator=lcz.Lanczos_Ops(D, d, self.Hs[i], self.Hs[i+1], self.L, self.R), k=lcz_k) for i in range(self.N-3)]
        
    def create_RL_ops(self, list_flag):
        self.R_boundary, self.L_boundary = self.RL_boundary_graph(self.H.right), self.RL_boundary_graph(self.H.left)
        if self.H_list_flag:
            self.R, self.L = [], []
            for i in range(self.N - 2):
                self.R.append(self.R_graph(self.H.mid[i]))
                self.L.append(self.L_graph(self.H.mid[i]))
        else:
            self.R = self.R_graph(self.H.mid)
            self.L = self.L_graph(self.H.mid)

        
    
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
        
    def RL_boundary_graph(self, Hi):
        x = tf.einsum('bij,cj->bci', Hi, self.plc.state)
        return tf.einsum('bci,ai->abc', x, tf.conj(self.plc.state))
    
    def R_graph(self, Hi):
        x = tf.einsum('abc,fcj->abfj', self.plc.R, self.plc.state)
        x = tf.einsum('ebij,abfj->aefi', Hi, x)
        return tf.einsum('adi,aefi->def', tf.conj(self.plc.state), x)
        #also changed the indices in einsum to take into account the dagger
    
    def L_graph(self, Hi):
        x = tf.einsum('def,fcj->decj', self.L, self.plc.state)
        x = tf.einsum('ebij,decj->dbci', Hi, x)
        return tf.einsum('adi,dbci->abc', tf.conj(self.plc.state), x)
        #also changed the indices in einsum to take into account the dagger

    