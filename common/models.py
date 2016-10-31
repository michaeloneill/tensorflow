import tensorflow as tf
from common.utility_fns import build_train_graph, get_loss

def build_mlp_model(params):

    ''' 
    build generic mlp model that can be adapted for regression or classification 
    
    '''

    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')


    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32,
                                           shape=(),
                                           name='dropout_keep_prob')

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')

    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, params['mlp'])
            
    with tf.name_scope('loss'):
        loss = get_loss(mlp_output, y_, params['train']['loss_fn']) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': mlp_output
    }
    
    return model



def build_cnn_model(params):

    ''' 
    build generic cnn model that can be adapted for regression or classification 
    
    '''

    
    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')


    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
            
    with tf.variable_scope('cnn'):
        cnn_output = build_cnn(x, dropout_keep_prob, params['cnn'])
            
    with tf.name_scope('loss'):
        loss = get_loss(cnn_output, y_, params['train']['loss_fn']) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': cnn_output
    }
    
    return model
