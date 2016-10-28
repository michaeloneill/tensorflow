import tensorflow as tf
from common.utility_fns import build_train_graph, get_loss


ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'identity': tf.identity
    }


############### GRAPHS #############################################

def build_cnn(x, dropout_keep_prob, params):

    nConvLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nConvLayers):

        kernel_size = params['kernel_size'][i]
        num_outputs = params['num_outputs'][i]
        layer_name = 'conv_layer'+str(i+1)
        act = params['activations'][i]
        
        if i==0: 
            inpt=x
        else:
            inpt = prev_layer

        with tf.variable_scope(layer_name) as scope:
            layer = tf.contrib.layers.convolution2d(
                inputs=inpt,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding='SAME',
                activation_fn=ACTIVATIONS[act],
                normalizer_fn=tf.contrib.layers.batch_norm)

            scope.reuse_variables()
            prev_layer = layer


    # flatten output of last conv_layer and pass to fc layers

    flattened_dim = prev_layer.get_shape()[1].value*prev_layer.get_shape()[2].value*prev_layer.get_shape()[3].value
    flattened = tf.reshape(prev_layer, [-1, flattened_dim])
    output = build_mlp(flattened, dropout_keep_prob, params['fc_params'])
        
    return output



def build_mlp(x, dropout_keep_prob, params):
    
    nLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nLayers):
            
        num_outputs = params['num_outputs'][i]
        layer_name = 'fc_layer'+str(i+1)
        act = params['activations'][i]
        
        if i==0:
            inpt = x
        else:
            inpt = prev_layer

        if params['dropout'][i]:
            
            with tf.name_scope('dropout'):
                inpt = tf.nn.dropout(inpt, dropout_keep_prob)
        
        with tf.variable_scope(layer_name) as scope:
            layer = tf.contrib.layers.fully_connected(
                inputs=inpt,
                num_outputs=num_outputs,
                activation_fn=ACTIVATIONS[act],
                normalizer_fn=tf.contrib.layers.batch_norm)

            scope.reuse_variables()
            prev_layer = layer

    return prev_layer


    
def build_dcnn(x, dropout_keep_prob, params):

    """ currently does not support pooling reversal """

    dmlp_output = build_mlp(x, dropout_keep_prob, params['fc_params'])
    rastered = tf.reshape(dmlp_output, [-1]+params['raster_shape'])
    
    nDeConvLayers = len(params['num_outputs'])
    prev_layer = None
    
    for i in range(nDeConvLayers):

        kernel_size = params['kernel_size'][i]
        num_outputs = params['num_outputs'][i]
        layer_name = 'deconv_layer'+str(i+1)
        act = params['activations'][i]
        
        if i==0: 
            inpt = rastered
        else:
            inpt = prev_layer

        with tf.variable_scope(layer_name) as scope:
            layer = tf.contrib.layers.convolution2d_transpose(
                inputs=inpt,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding='SAME',
                activation_fn=ACTIVATIONS[act],
                normalizer_fn=tf.contrib.layers.batch_norm)
            
            scope.reuse_variables()
            prev_layer = layer

    return prev_layer


############################## MODELS ##########################################


def build_mlp_model(params):

    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')


    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                
    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, params['mlp'])
            
    with tf.name_scope('loss'):
        loss = get_loss(mlp_output, y_, params['train']['loss_fn']) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': mlp_output
    }
    
    return model


def build_cnn_model(params):

    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')


    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                
    with tf.variable_scope('cnn'):
        cnn_output = build_cnn(x, dropout_keep_prob, params['cnn'])
            
    with tf.name_scope('loss'):
        loss = get_loss(cnn_output, y_, params['train']['loss_fn']) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': cnn_output
    }
    
    return model


