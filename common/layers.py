import tensorflow as tf
import numpy as np
import pdb


ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax
    }


def convPoolLayer(inpt, filtr_shape, layer_name, act='sigmoid', pool=True):

    """
 
    filtr_shape is [fHeight, fWidth, nInputFeatureMaps, nOutputFeatureMaps]
    inpt is b x imWidth x imHight x nChannels

    """
    
    with tf.name_scope(layer_name):
        with tf.name_scope('filtr'):
            filtr = weight_variable(filtr_shape)
        with tf.name_scope('biases'):
            biases = bias_variable([filtr_shape[-1]])
        with tf.name_scope('convolved'):
            convolved = ACTIVATIONS[act](conv2d(inpt, filtr)+biases)

        if pool:
            with tf.name_scope('pooled'):
                pooled = max_pool_2x2(convolved)
                return pooled, filtr
        else:
            return convolved, filtr 

    
def fullyConnectedLayer(inpt, inpt_dim, outpt_dim, layer_name,
                        act='sigmoid', weights=None):

    with tf.name_scope(layer_name):
        if weights is None:
            with tf.name_scope('weights'):
                weights = weight_variable([inpt_dim, outpt_dim])
        with tf.name_scope('biases'):
            biases = bias_variable([outpt_dim])
        with tf.name_scope('activations'):
            activations = ACTIVATIONS[act](tf.matmul(inpt, weights)+biases)
        return activations, weights


def deConvPoolLayer(inpt, filtr, outpt_shape, layer_name, act='sigmoid'):

    """ 
    Currently does not support de-pooling
    inpt is b x H x W x nInputFeatureMaps
    outpt_shape is [b, H, W, nOutputFeatureMaps]
    filtr is filtr_dim x filter_dim x nInputFeatureMaps x nOutputFeatureMaps

    """
    with tf.name_scope(layer_name):

        with tf.name_scope('biases'):
            biases = bias_variable([outpt_shape[-1]])
        with tf.name_scope('deconvolved'):
            deconvolved = ACTIVATIONS[act](deconv2d(inpt, filtr, outpt_shape) + biases)
        return deconvolved


def weight_variable(shape):
    return tf.get_variable('weights', shape, tf.float32, tf.random_normal_initializer())

def bias_variable(shape):
    return tf.get_variable('biases', shape, tf.float32, tf.constant_initializer(0.1))


def conv2d(inpt, W):
    return tf.nn.conv2d(inpt, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(inpt):

    return tf.nn.max_pool(inpt, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding= 'SAME')

def deconv2d(inpt, filtr, output_shape):
    return tf.nn.conv2d_transpose(inpt, filtr, output_shape, strides=[1,1,1,1],
                          padding="SAME") 
