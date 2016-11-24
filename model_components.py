import tensorflow as tf
from layers import fullyConnectedLayer, convPoolLayer
from lstm_cells import BNLSTMCell, ConvLSTMCell, BNConvLSTMCell
import numpy as np

ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'identity': tf.identity
    }


def build_rnn(x, dropout_keep_prob, is_training, params):

    with tf.name_scope('cell'):
        if params['cell_type'] == 'BasicLSTM':
            cell = tf.nn.rnn_cell.BasicLSTMCell(
                params['dim_hidden'], forget_bias=1.0, state_is_tuple=True)
        elif params['cell_type'] == 'BNLSTM':
            cell = BNLSTMCell(
                params['dim_hidden'], is_training,
                forget_bias=1.0)  # state_is_tuple assumed True
        elif params['cell_type'] == 'ConvLSTM':
            cell = ConvLSTMCell(
                params['shape'], params['filter_size'],
                params['num_output_feature_maps'],
                forget_bias=1.0)  # state_is_tuple assumed True
        elif params['cell_type'] == 'BNConvLSTMCell':
            cell = BNConvLSTMCell(
                params['shape'], params['filter_size'],
                params['num_output_feature_maps'],
                is_training, forget_bias=1.0)  # state_is_tuple assumed True
        else:
            raise ValueError('Cell type not recognised')
        
        if params['dropout']:
            with tf.name_scope('dropped_cell'):
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, input_keep_prob=dropout_keep_prob)

        # with tf.name_scope('stacked_lstm'):
        #     cell = tf.nn.rnn_cell.MultiRNNCell(
        # [cell]*params['num_layers'], state_is_tuple=True)
        
        b_size = tf.shape(x)[0]
        with tf.name_scope('init_state'):
            state = cell.zero_state(b_size, tf.float32)

    with tf.name_scope('hiddens'):
        hiddens = [None]*params['seq_len']
        for t in range(params['seq_len']):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            hiddens[t], state = cell(x[:, t], state)  # hiddens is s * [b x h]

    return hiddens, state


def hidden_to_output(hiddens, dim_output, activation=tf.identity):

    """
    To be used on hidden state of rnn.
    hiddens can be s * [bxh] or [bxh] (fully connected cell layers)
    or it can be s*[bxHxWxnum_output_feature_maps] or
    [bxHxWxnum_output_feature_maps] (convolutional layers)
    Either way
      output (s*b) x dim_output or b*dim_output
    """

    if type(hiddens) is list:
        hiddens = tf.concat(0, hiddens)
    if len(hiddens.get_shape().as_list()) == 4:  # convolutional cell
        dim_hidden = np.prod(hiddens.get_shape().as_list()[1:])
        hiddens = tf.reshape(hiddens, [-1, dim_hidden])
    else:
        assert len(hiddens.get_shape().as_list()) == 2
        dim_hidden = hiddens.get_shape()[-1].value

    stdev = 1.0/np.sqrt(dim_output)
    W_out = tf.Variable(tf.random_uniform(
        [dim_hidden, dim_output], -stdev, stdev))

    return activation(tf.matmul(hiddens, W_out))
                                       

def build_cnn(x, dropout_keep_prob, is_training, params):

    nConvLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nConvLayers):

        kernel_size = params['kernel_size'][i]
        num_outputs = params['num_outputs'][i]
        layer_name = 'conv_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        pool = params['pool'][i]
        
        if i == 0:
            inpt = x
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
    flattened_dim = np.prod(prev_layer.get_shape().as_list()[1:])
    flattened = tf.reshape(prev_layer, [-1, flattened_dim])

    output = build_mlp(flattened, dropout_keep_prob,
                       is_training, params['fc_params'])
        
    return output


def build_mlp(x, dropout_keep_prob, is_training, params):
    
    nLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nLayers):
            
        num_outputs = params['num_outputs'][i]
        layer_name = 'fc_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        
        if i == 0:
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
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), params['grad_clip'])

        optimizer = tf.train.MomentumOptimizer(
            params['learning_rate'], momentum=params['momentum'])
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        tf.add_to_collection('train_op', train_op)

        return train_op


def get_loss(logits, targets, loss_fn='mean_squared'):
    if loss_fn == 'mean_squared':
        return tf.reduce_sum(tf.square(logits-targets))
    elif loss_fn == 'sigmoid_cross_entropy_with_logits':
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits, targets))
    elif loss_fn == 'softmax_cross_entropy_with_logits':
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, targets))
    else:
        raise ValueError('unrecognised loss function')

    
# def build_dcnn(x, dropout_keep_prob, params):

#     """ Currently does not support pooling reversal """

#     dmlp_output = build_mlp(x, dropout_keep_prob, params['fc_params'])
#     rastered = tf.reshape(dmlp_output, [-1]+params['raster_shape'])
    
#     nDeConvLayers = len(params['num_outputs'])
#     prev_layer = None
    
#     for i in range(nDeConvLayers):

#         kernel_size = params['kernel_size'][i]
#         num_outputs = params['num_outputs'][i]
#         layer_name = 'deconv_layer'+str(i+1)
#         act = params['activations'][i]
        
#         if i == 0:
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
