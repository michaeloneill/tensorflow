import tensorflow as tf
from model_components import build_cnn, build_mlp, build_rnn
from model_components import build_train_graph, get_loss, hidden_to_output


def build_mlp_model(params):

    '''
    Build generic mlp model that
    can be adapted for regression or classification
    '''
    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')
        tf.add_to_collection('x', x)

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')
        tf.add_to_collection('y_', y_)

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32,
                                           shape=(),
                                           name='dropout_keep_prob')
        tf.add_to_collection('dropout_keep_prob', dropout_keep_prob)

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
        tf.add_to_collection('is_training', is_training)

    with tf.variable_scope('mlp'):
        output = build_mlp(x, dropout_keep_prob, params['mlp'])
        tf.add_to_collection('output', output)
            
    with tf.name_scope('loss'):
        loss = get_loss(output, y_, params['train']['loss_fn'])
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': output
    }
    
    return model


def build_cnn_model(params):

    '''
    Build generic cnn model that can be adapted for
    regression or classification
    '''
    
    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')
        tf.add_to_collection('x', x)

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')
        tf.add_to_collection('y_', y_)

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')
        tf.add_to_collection('dropout_keep_prob', dropout_keep_prob)

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
        tf.add_to_collection('is_training', is_training)
    with tf.variable_scope('cnn'):
        output = build_cnn(x, dropout_keep_prob, params['cnn'])
        tf.add_to_collection('output', output)
            
    with tf.name_scope('loss'):
        loss = get_loss(output, y_, params['train']['loss_fn'])
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': output
    }
    
    return model


def build_rnn_model(params):

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')
        tf.add_to_collection('x', x)
        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')
        tf.add_to_collection('y_', y_)

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32,
                                           name='dropout_keep_prob')
        tf.add_to_collection('dropout_keep_prob', dropout_keep_prob)

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
        tf.add_to_collection('is_training', is_training)
    
    with tf.name_scope('lstm'):
        hiddens, _ = build_rnn(x, dropout_keep_prob,
                               is_training, params['rnn'])
            
    with tf.name_scope('output'):
        output = hidden_to_output(
            hiddens, params['inpt_shape']['y_'][1],
            params['rnn']['out_activation'])  # (s*b) x dim_output
        tf.add_to_collection('output', output)
        
    with tf.name_scope('loss'):
        loss = get_loss(output, y_, params['train']['loss_fn'])
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': output,
    }

    return model
                        








