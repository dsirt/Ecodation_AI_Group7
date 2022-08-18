# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 22:44:31 2022

@author: ITU
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data", one_hot = True)

import tensorflow as tf
import matplotlib.pyplot as plt

#Parametreler
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

#tf Graph inputs
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#Model oluşturma
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)

#Minimizasyon işlemleri
cross_entropy = y*tf.log(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Çizim ayarları
avg_set = []
epoch_set = []

#değişkenlerin ilk atamalarını yapma işlemi
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            
            
        #her bir epoch için logları gösterme
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost:","{:.9f}".format(avg_cost))
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
        
    print("Eğitim fazı tamamlandı")
    
    plt.plot(epoch_set, avg_set, 'o', label = 'Logistic Regression Eğitim Fazı')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    #Modelimizi test ediyoruz
    dogru_tahmin = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    #accuracy (başarım) hesabı
    accuracy = tf.reduce_mean(tf.cast(dogru_tahmin, "float"))
    print("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    
    
    
    
    
    
    






