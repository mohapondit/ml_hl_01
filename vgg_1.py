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