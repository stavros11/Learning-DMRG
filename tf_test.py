# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:15:15 2018

@author: Admin
"""

import tensorflow as tf

x = tf.Variable(1., dtype=tf.float32)
y = tf.constant(2., dtype=tf.float32)

loss = 0
for p in range(2, 6):
    loss += tf.pow(x, p)

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(200):
        print(sess.run(x), sess.run(loss))
        sess.run(train)
        