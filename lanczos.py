# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:48:28 2018

@author: Admin
"""

import tensorflow as tf

def lanczos_algorithm(operator, B_init, k):
    # The diagonal elements of the tridiagonal matrix
    alpha = tf.Variable(tf.zeros(shape=(k,), dtype=tf.complex64))
    # The off-diagonal elements of the triagonal matrix
    beta = tf.Variable(tf.zeros(shape=(k-1,), dtype=tf.complex64))
    # The matrix V used to transform the eigenvectors
    vector = tf.Variable(tf.zeros(shape=(k,) + tuple(B_init.get_shape().as_list()), dtype=tf.complex64))
    ## Note: to transform the eigenvectors of the tridiagonal, sum over the first index of V
    ## which has dimension k
    
    vector[0].assign(B_init)
    w = operator.apply(vector[0])
    alpha[0].assign(contract_vectors(w, B_init))
    w = w - alpha[0] * B_init
    
    for i in range(1, k):
        beta[i-1].assign(contract_vectors(w, w))
        vector[i].assign(vector[i-1] / beta[i-1])
        
        w = operator.apply(vector[i])
        alpha[i].assign(contract_vectors(w, vector[i]))
        w = w - alpha[i] * vector[i] - beta[i-1] * vector[i-1]
        
    return vector, alpha, beta
        
def contract_vectors(up, down):
    ## Assumes same order of indices in both Bs
    return tf.reduce_sum(tf.multiply(tf.conj(up), down))

class Lanczos_OperatorM(object):
    def __init__(self, H1, H2, L, R):
        self.H1, self.H2 = H1, H2
        self.L, self.R = L, R
        
    def apply(self, B):
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        return tf.einsum('afgij,efg->aeij', LB, self.R)
        ## Final indices in order DL, DR, d, d
    
class Lanczos_Operator0(object):
    def __init__(self, H1, H2, LR):
        self.H1, self.H2 = H1, H2
        self.LR = LR
        
    def apply(self, B):
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,dbkj->adik', LB, self.H2)
        return tf.einsum('adik,dli->alk', LB, self.H1)
        ## Final indices in order D, d, d
    
class Lanczos_OperatorN(Lanczos_Operator0):
    def apply(self, B):
        LB = tf.einsum('abc,cij->abij', self.LR, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        return tf.einsum('adkj,dlj->akl', LB, self.H2)
        ## Final indices in order D, d, d