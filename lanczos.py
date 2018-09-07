# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:48:28 2018

@author: Admin
"""

import tensorflow as tf

def lanczos_algorithm(operator, B_init, k):
    # The diagonal elements of the tridiagonal matrix
    alpha = tf.Variable(tf.zeros(shape=(k,)))
    # The off-diagonal elements of the triagonal matrix
    beta = tf.Variable(tf.zeros(shape=(k-1,)))
    # The matrix V used to transform the eigenvectors
    vector = tf.Variable(tf.zeros(shape=(k,) + tuple(B_init.get_shape().as_list())))
    ## Note: to transform the eigenvectors of the tridiagonal, sum over the first index of V
    ## which has dimension k
    
    v_old = B_init
    vector[0] = B_init
    
    w = operator.apply(B_init)
    alpha[0] = contract_vectors(w, B_init)
    w = w - alpha[0] * B_init
    
    for i in range(1, k):
        beta[i-1] = contract_vectors(w, w)
        v_new = v_old / beta[i]
        vector[i] = v_new
        
        w = operator.apply(v_new)
        alpha[i] = contract_vectors(w, v_new)
        w = w - alpha[i] * v_new - beta[i-1] * v_old
        v_old = v_new
        
    return vector, alpha, beta
        
def contract_vectors(up, down):
    ## Assumes same order of indices in both Bs
    return tf.reduce_sum(tf.multiply(tf.conj(up), down))

class Lanczos_OperatorM(object):
    def __init__(self, DL, DR, H1, H2, L, R):
        self.DL, self.DR = DL, DR
        self.d = int(H1.shape[-1])
        self.dims = DL * DR * self.d * self.d
        self.shape = [self.dims, self.dims]
        self.dtype = tf.complex64

        self.H1, self.H2 = H1, H2
        self.L, self.R = L, R
        
    def apply(self, B):
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        return tf.einsum('afgij,efg->aeij', LB, self.R)
        ## Final indices in order DL, DR, d, d
    
    ## Fixed indices to apply adjoint a-c, i-k, j-l, e-g
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.DL, self.DR, self.d, self.d))
        LB = tf.einsum('cba,cgkl->abgkl', tf.conj(self.L), B)
        LB = tf.einsum('abgkl,bdki->adgil', LB, tf.conj(self.H1))
        LB = tf.einsum('adgil,dflj->afgij', LB, tf.conj(self.H2))
        LB = tf.einsum('afgij,gfe->aeij', LB, tf.conj(self.R))
        ## Final indices in order DL, d, d, DR
        return tf.reshape(LB, shape=(self.dims,))
    
class Lanczos_Operator0(object):
    def __init__(self, H1, H2, LR):
        self.d = int(H1.shape[-1])
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
    
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        LB = tf.einsum('cba,cij->abij', tf.conj(self.LR), B)
        LB = tf.einsum('abij,bdik->adkj', LB, tf.conj(self.H2))
        LB = tf.einsum('adkj,djl->akl', LB, tf.conj(self.H1))
        ## Final indices in the order of the TN-graph
        return tf.reshape(LB, shape=(self.dims,))
    
class Lanczos_OperatorN(Lanczos_Operator0):
    def apply(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        LB = tf.einsum('adkj,dlj->akl', LB, self.H2)
        ## Final indices in the order of the TN-graph
        return tf.reshape(LB, shape=(self.dims,))
    
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.d, self.d, self.d))
        LB = tf.einsum('cba,cij->abij', tf.conj(self.LR), B)
        LB = tf.einsum('abij,bdik->adkj', LB, tf.conj(self.H1))
        LB = tf.einsum('adkj,djl->akl', LB, tf.conj(self.H2))
        ## Final indices in the order of the TN-graph
        return tf.reshape(LB, shape=(self.dims,))