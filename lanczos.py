# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:40:09 2018

@author: Admin
"""

import tensorflow as tf

class Lanczos_Ops(object):
    def __init__(self, D, d, H1, H2, L, R):
        self.D, self.d = D, d
        self.dims = D*D*d*d
        self.shape = [self.dims, self.dims]
        self.dtype = tf.complex64
        
        self.H1, self.H2 = H1, H2
        self.L, self.R = L, R
        
    def apply(self, x):
        B = tf.reshape(x, shape=(self.D, self.D, self.d, self.d))
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        LB = tf.einsum('afgij,efg->aeij', LB, self.R)
        return tf.reshape(LB, shape=(self.dims,))
        
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.D, self.D, self.d, self.d))
        B = tf.conj(tf.transpose(B, [1, 0, 3, 2]))
        LB = tf.einsum('abc,cgkl->abgkl', self.L, B)
        LB = tf.einsum('abgkl,bdik->adgil', LB, self.H1)
        LB = tf.einsum('adgil,dfjl->afgij', LB, self.H2)
        LB = tf.einsum('afgij,efg->aeij', LB, self.R)
        return tf.conj(tf.reshape(tf.transpose(LB, [1, 0, 3, 2]), shape=(self.dims,)))
    
class Lanczos_Ops_Boundary(object):
    def __init__(self, D, d, H1, H2, LR):
        self.D, self.d = D, d
        self.dims = D*d*d
        self.shape = [self.dims, self.dims]
        self.dtype = tf.complex64
        
        self.H1, self.H2 = H1, H2
        self.LR = LR
        
    def apply(self, x):
        B = tf.reshape(x, shape=(self.D, self.D, self.d, self.d))
        LB = tf.einsum('abc,cij->abij', self.L, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        LB = tf.einsum('adkj,dlj->akl', LB, self.H2)
        return tf.reshape(LB, shape=(self.dims,))
        
    def apply_adjoint(self, x):
        B = tf.reshape(x, shape=(self.D, self.D, self.d, self.d))
        B = tf.conj(tf.transpose(B, [1, 0, 3, 2]))
        LB = tf.einsum('abc,cij->abij', self.L, B)
        LB = tf.einsum('abij,bdki->adkj', LB, self.H1)
        LB = tf.einsum('adkj,dlj->akl', LB, self.H2)
        return tf.conj(tf.reshape(tf.transpose(LB, [1, 0, 3, 2]), shape=(self.dims,)))