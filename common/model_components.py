import tensorflow as tf
from layers import fullyConnectedLayer, convPoolLayer
import pdb

ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'identity': tf.identity
    }


def build_cnn(x, dropout_keep_prob, is_training, params):

    nConvLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nConvLayers):

        kernel_size = params['kernel_size'][i]
        num_outputs = params['num_outputs'][i]
        layer_name = 'conv_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        pool = params['pool'][i]
        
        if i==0: 
            inpt=x
        else:
            inpt = prev_layer


        layer = convPoolLayer(
            inputs=inpt,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=1,
            padding='SAME',
            layer_name=layer_name,
            is_training=is_training,
            activation_fn=act,
            pool=pool
        )

        prev_layer = layer


    # flatten output of last conv_layer and pass to fc layers

    flattened_dim = prev_layer.get_shape()[1].value*prev_layer.get_shape()[2].value*prev_layer.get_shape()[3].value
    flattened = tf.reshape(prev_layer, [-1, flattened_dim])
    output = build_mlp(flattened, dropout_keep_prob, is_training, params['fc_params'])
        
    return output



def build_mlp(x, dropout_keep_prob, is_training, params):
    
    nLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nLayers):
            
        num_outputs = params['num_outputs'][i]
        layer_name = 'fc_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        
        if i==0:
            inpt = x
        else:
            inpt = prev_layer

        if params['dropout'][i]:
            with tf.name_scope('dropout'):
                inpt = tf.nn.dropout(inpt, dropout_keep_prob)

        layer = fullyConnectedLayer(
            inputs=inpt,
            num_outputs=num_outputs,
            layer_name=layer_name,
            is_training=is_training,
            activation_fn=act
        )
        prev_layer = layer

    return prev_layer



def build_train_graph(loss, params):

    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), params['grad_clip'])

        optimizer = tf.train.MomentumOptimizer(params['learning_rate'], momentum=params['momentum'])
        train_step = optimizer.apply_gradients(zip(grads, tvars))

        return train_step


def get_loss(logits, targets, loss_fn='mean_squared'):
    if loss_fn == 'mean_squared':
        return tf.reduce_sum(tf.square(logits-targets)) # squared error over minibatch
    elif loss_fn == 'sigmoid_cross_entropy_with_logits':
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, targets))
    elif loss_fn == 'softmax_cross_entropy_with_logits':
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, targets))
    else:
        print 'unrecognised loss function'
        sys.exit(2)







# def build_dcnn(x, dropout_keep_prob, params):

#     """ currently does not support pooling reversal """

#     dmlp_output = build_mlp(x, dropout_keep_prob, params['fc_params'])
#     rastered = tf.reshape(dmlp_output, [-1]+params['raster_shape'])
    
#     nDeConvLayers = len(params['num_outputs'])
#     prev_layer = None
    
#     for i in range(nDeConvLayers):

#         kernel_size = params['kernel_size'][i]
#         num_outputs = params['num_outputs'][i]
#         layer_name = 'deconv_layer'+str(i+1)
#         act = params['activations'][i]
        
#         if i==0: 
#             inpt = rastered
#         else:
#             inpt = prev_layer

#         with tf.variable_scope(layer_name) as scope:
#             layer = tf.contrib.layers.convolution2d_transpose(
#                 inputs=inpt,
#                 num_outputs=num_outputs,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding='SAME',
#                 activation_fn=ACTIVATIONS[act],
#                 normalizer_fn=tf.contrib.layers.batch_norm)
            
#             scope.reuse_variables()
#             prev_layer = layer

#     return prev_layer
