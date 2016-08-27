""" a vgg net for bangla handwritten character recognition

by Uzzal Podder,
uzzal@bsmrstu.edu.bd

 """


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
## _------------------------------------------------------------------

###############################################
##         Data preprocessing

import tflearn.datasets.mnist as mnist
X1, Y1, testX1, testY1 = mnist.load_data(one_hot=True)
X = X1.reshape([-1, 64, 64, 1])
testX = testX1.reshape([-1, 64, 64, 1])


Y = Y1.reshape([-1, 64, 64, 1])
testY = testY1.reshape([-1, 64, 64, 1])


#########################################
#          Building 'VGG Network'

network = input_data(shape=[None, 64, 64, 1])


network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2, strides=2)



network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)


network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)


network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)


network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 300, activation='softmax')


network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)

## _____________ MODEL END __________________________________



############################################################
##          Traingng

model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
"""
model.fit({'input': X}, {'target': Y}, n_epoch=4,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
"""

model.save('mnist_vgg.tflearn')



#network = local_response_normalization(network)