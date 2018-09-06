# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:40:09 2018

@author: Admin
"""

import tensorflow as tf
   
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
        self.plc = Placeholders()
        ## Create hamiltonian object
        self.H = Hamiltonian(H0, Hs, HN)
        
        ## Determine if Hamiltonian is list or same for all middle sites
        self.H_list_flag = (len(Hs.shape) >= 5)
        self.create_RL_ops()
        self.create_lanczos_ops(lcz_k=lcz_k)
        
    def create_RL_ops(self):
        self.R_boundary, self.L_boundary = self.RL_boundary_graph(self.H.right), self.RL_boundary_graph(self.H.left)
        
        if self.H_list_flag:
            self.R, self.L = [], []
            for i in range(self.N - 2):
                self.R.append(self.R_graph(self.H.mid[i]))
                self.L.append(self.L_graph(self.H.mid[i]))
        else:
            self.R = self.R_graph(self.H.mid)
            self.L = self.L_graph(self.H.mid)
            
    def create_lanczos_ops(self, lcz_k):
        if self.H_list_flag:
            self.lanczos0 = tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_Operator0(self.H.left, self.H.mid[0], self.plc.R), 
                    k=lcz_k, name="lanczos_bidiag_left")
            self.lanczosN = tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_Ops_Boundary(self.H.mid[-1], self.H.right, self.plc.L), 
                    k=lcz_k, name="lanczos_bidiag_right")
            
            self.lanczosM = [tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_OperatorM(self.D[i-1], self.D[i+1], self.H.mid[i], self.H.mid[i+1], 
                                               self.plc.L, self.plc.R), 
                    k=lcz_k, name="lanczos_bidiag_mid%d"%i) for i in range(self.N - 3)]
        
        else:
            self.lanczos0 = tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_Operator0(self.H.left, self.H.mid, self.plc.R), 
                    k=lcz_k, name="lanczos_bidiag_left")
            self.lanczosN = tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_Ops_Boundary(self.H.mid, self.H.right, self.plc.L), 
                    k=lcz_k, name="lanczos_bidiag_right")
            
            self.lanczosM = [tf.contrib.solvers.lanczos.lanczos_bidiag(
                    operator=Lanczos_OperatorM(self.D[i-1], self.D[i+1], self.H.mid, self.H.mid, 
                                               self.plc.L, self.plc.R), 
                    k=lcz_k, name="lanczos_bidiag_mid%d"%i) for i in range(self.N - 3)]
        
    def RL_boundary_graph(self, Hi):
        x = tf.einsum('bij,cj->bci', Hi, self.plc.state)
        return tf.einsum('bci,ai->abc', x, tf.conj(self.plc.state))
    
    def R_graph(self, Hi):
        x = tf.einsum('abc,fcj->abfj', self.plc.R, self.plc.state)
        x = tf.einsum('ebij,abfj->aefi', Hi, x)
        return tf.einsum('dai,aefi->def', tf.conj(self.plc.state), x)
    
    def L_graph(self, Hi):
        x = tf.einsum('def,fcj->decj', self.plc.L, self.plc.state)
        x = tf.einsum('ebij,decj->dbci', Hi, x)
        return tf.einsum('dai,dbci->abc', tf.conj(self.plc.state), x)
        #also changed the indices in einsum to take into account the dagger

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

class Lanczos_OperatorM(object):
    def __init__(self, DL, DR, H1, H2, L, R):
        self.DL, self.DR = DL, DR
        self.d = H1.shape[-1]
        self.dims = DL * DR * self.d * self.d
        self.shape = [self.dims, self.dims]
        self.dtype = tf.complex64

        self.H1, self.H2 = H1, H2
        self.L, self.R = L, R
        
    def apply(self, x):
        B = tf.reshape(x, shape=(self.DL, self.DR, self.d, self.d))
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        LB = tf.einsum('afgij,efg->aije', LB, self.R)
        ## Final indices in order DL, d, d, DR
        return tf.reshape(LB, shape=(self.dims,))
    
    ### CHECK APPLY ADJOINT! ###
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.D, self.D, self.d, self.d))
        B = tf.conj(tf.transpose(B, [1, 0, 3, 2]))
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        LB = tf.einsum('afgij,efg->aeij', LB, self.R)
        return tf.conj(tf.reshape(tf.transpose(LB, [1, 0, 3, 2]), shape=(self.dims,)))
    
class Lanczos_Operator0(object):
    def __init__(self, H1, H2, LR):
        self.d = H1.shape[-1]
        self.dims = self.d**3
        self.shape = [self.dims, self.dims]
        self.dtype = tf.complex64
        
        self.H1, self.H2 = H1, H2
        self.LR = LR
        
    def apply(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,dbkj->adik', LB, self.H2)
        LB = tf.einsum('adik,dli->lka', LB, self.H1)
        ## Final indices in the order of the TN-graph
        return tf.reshape(LB, shape=(self.dims,))
    
    ### CHECK APPLY ADJOINT! ###
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        B = tf.conj(tf.transpose(B, [1, 0, 3, 2]))
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        LB = tf.einsum('adkj,dlj->akl', LB, self.H2)
        return tf.conj(tf.reshape(tf.transpose(LB, [1, 0, 3, 2]), shape=(self.dims,)))
    
class Lanczos_OperatorN(Lanczos_Operator0):
    def apply(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        LB = tf.einsum('adkj,dlj->akl', LB, self.H1)
        ## Final indices in the order of the TN-graph
        return tf.reshape(LB, shape=(self.dims,))
    
    ### ADD ADJOINT! ###
        
    
