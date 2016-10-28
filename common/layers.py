import tensorflow as tf
import numpy as np
import pdb

ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'identity': tf.identity
    }


def batch_norm_wrapper(inputs, is_training, layer_name, decay = 0.999, epsilon=1e-3):

    with tf.name_scope(layer_name):
        with tf.name_scope('bn_scale'):
            scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        with tf.name_scope('bn_beta'):
            beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        with tf.name_scope('pop_mean'):
            pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        with tf.name_scope('pop_var'):
            pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            train_time = tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)

        test_time = tf.nn.batch_normalization(inputs,
                            pop_mean, pop_var, beta, scale, epsilon)

        return is_training*train_time + (1-is_training)*test_time

def convPoolLayer(inpt, filtr_shape, layer_name, is_training, act='sigmoid', pool=True):

    """
 
    filtr_shape is [fHeight, fWidth, nInputFeatureMaps, nOutputFeatureMaps]
    inpt is b x imWidth x imHight x nChannels

    """
    
    with tf.name_scope(layer_name):
        with tf.name_scope('filtr'):
            filtr = weight_variable(filtr_shape)
        with tf.name_scope('logits'):
            logits = conv2d(inpt, filtr)
        with tf.name_scope('logits_bn'):
            logits_bn = batch_norm_wrapper(logits, is_training, layer_name)
        with tf.name_scope('activations'):
            activations = ACTIVATIONS[act](logits_bn)
        if pool:
            with tf.name_scope('pooled'):
                pooled = max_pool_2x2(activations)
                return pooled
        else:
            return activations

    
def fullyConnectedLayer(inpt, num_outputs, layer_name,
                        is_training,
                        act='sigmoid'):

    with tf.name_scope(layer_name):

        with tf.variable_scope('weights'):
            weights = weight_variable([inpt.get_shape()[-1].value, num_outputs])
        with tf.name_scope('logits'):
            logits = tf.matmul(inpt, weights)
        with tf.name_scope('logits_bn'):
            logits_bn = batch_norm_wrapper(logits, is_training, layer_name)
        with tf.name_scope('activations'):
            activations = ACTIVATIONS[act](logits_bn)

        return activations


def weight_variable(shape):
    
    return tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(), dtype=tf.float32)



def conv2d(inpt, W):
    return tf.nn.conv2d(inpt, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(inpt):

    return tf.nn.max_pool(inpt, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding= 'SAME')

def deconv2d(inpt, filtr, output_shape):
    return tf.nn.conv2d_transpose(inpt, filtr, output_shape, strides=[1,1,1,1],
                          padding="SAME") 



# def deConvPoolLayer(inpt, filtr, outpt_shape, layer_name, act='sigmoid'):

#     """ 
#     Currently does not support de-pooling
#     inpt is b x H x W x nInputFeatureMaps
#     outpt_shape is [b, H, W, nOutputFeatureMaps]
#     filtr is filtr_dim x filter_dim x nInputFeatureMaps x nOutputFeatureMaps

#     """
#     with tf.name_scope(layer_name):

#         with tf.name_scope('biases'):
#             biases = bias_variable([outpt_shape[-1]])
#         with tf.name_scope('deconvolved'):
#             deconvolved = ACTIVATIONS[act](deconv2d(inpt, filtr, outpt_shape) + biases)
#         return deconvolved


# def bias_variable(shape):
#     return tf.get_variable('biases', shape, tf.float32, tf.constant_initializer(0.1))
