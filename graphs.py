# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:40:09 2018

@author: Admin
"""

import tensorflow as tf
import lanczos as lcz
   
class Operations(object):
    def __init__(self, D, H0, Hs, HN, lcz_k):
        ### Hamiltonian in MPO form ###
        ## DH: Hidden dimension for Hamiltonian
        ## H0: Hamiltonian for first site: Complex (DH, d, d)
        ## Hs: Hamiltonian for middle chain: Complex (N-2, DH, DH, d, d)
        ## HN: Hamiltonian for last site: Complex (DH, d, d)
        self.D = D        
        self.N = len(Hs) + 2
                
        ## Create placeholders object
        self.plc = Placeholders(d=H0.shape[-1], D=D, DH=H0.shape[0])
        ## Create hamiltonian object
        self.H = Hamiltonian(H0, Hs, HN)
        
        ## Determine if Hamiltonian is list or same for all middle sites
        #self.H_list_flag = (len(Hs.shape) >= 5)
        #Make H_list_flag always true because of the plc.state problems 
        #(have to define dimension --> different placeholder for each state)
        self.create_RL_ops()
        self.create_lanczos_ops(lcz_k=lcz_k)
        
    def create_RL_ops(self):
        self.L = [self.RL_boundary_graph(self.H.left, self.plc.state[0])]
        self.R = []
        for i in range(self.N - 3):
            self.R.append(self.R_graph(i))
            self.L.append(self.L_graph(i))
            
        self.R.append(self.RL_boundary_graph(self.H.right, self.plc.state[0]))
            
    def create_lanczos_ops(self, lcz_k):
        self.lanczos0 = lcz.lanczos_algorithm(
                operator=lcz.Lanczos_Operator0(self.H.left, self.H.mid[0], self.plc.R[0]), 
                B_init=self.B_boundary0_graph(), k=lcz_k)
        
        self.lanczosN = lcz.lanczos_algorithm(
                operator=lcz.Lanczos_OperatorN(self.H.mid[-1], self.H.right, self.plc.L[-1]), 
                B_init=self.B_boundaryN_graph(), k=lcz_k)
            
        self.lanczosM = [lcz.lanczos_algorithm(
                operator=lcz.Lanczos_OperatorM(self.H.mid[i], self.H.mid[i+1], self.plc.L[i], self.plc.R[i+1]), 
                B_init=self.B_graph(i+1), k=lcz_k) for i in range(self.N - 3)]
        
    def RL_boundary_graph(self, Hi, s):
        x = tf.einsum('bij,cj->bci', Hi, s)
        return tf.einsum('bci,ai->abc', x, tf.conj(s))
    
    def R_graph(self, i):
        x = tf.einsum('abc,fcj->abfj', self.plc.R[i+1], self.plc.state[i+2])
        x = tf.einsum('ebij,abfj->aefi', self.H.mid[i+1], x)
        return tf.einsum('dai,aefi->def', tf.conj(self.plc.state[i+2]), x)
    
    def L_graph(self, i):
        x = tf.einsum('def,fcj->decj', self.plc.L[i], self.plc.state[i+1])
        x = tf.einsum('ebij,decj->dbci', self.H.mid[i], x)
        return tf.einsum('dai,dbci->abc', tf.conj(self.plc.state[i+1]), x)
    
    def B_graph(self, i):
        return tf.einsum('abi,bcj->acij', self.plc.state[i], self.plc.state[i+1])
    
    def B_boundary0_graph(self):
        return tf.einsum('ai,abj->bij', self.plc.state[0], self.plc.state[1])
    
    def B_boundaryN_graph(self):
        return tf.einsum('abi,bj->aij', self.plc.state[-1], self.plc.state[0])

class Placeholders(object):
    def __init__(self, d, D, DH):
        self.state = [tf.placeholder(dtype=tf.complex64, shape=(d, d))]
        self.R = []
        self.L = [tf.placeholder(dtype=tf.complex64, shape=(D[0], DH, D[0]))]
        for i in range(1, len(D)-1):
            self.state.append(tf.placeholder(dtype=tf.complex64, shape=(D[i-1], D[i], d)))
            self.R.append(tf.placeholder(dtype=tf.complex64, shape=(D[i], DH, D[i])))
            self.L.append(tf.placeholder(dtype=tf.complex64, shape=(D[i], DH, D[i])))
        
        self.state.append(tf.placeholder(dtype=tf.complex64, shape=(D[-2], D[-1], d)))
        self.R.append(tf.placeholder(dtype=tf.complex64, shape=(D[-1], DH, D[-1])))
        
class Hamiltonian(object):
    def __init__(self, H0, Hs, HN):
        self.left = tf.constant(H0, dtype=tf.complex64, shape=H0.shape)
        self.mid = tf.constant(Hs, dtype=tf.complex64, shape=Hs.shape)
        self.right = tf.constant(HN, dtype=tf.complex64, shape=HN.shape)
        
    
