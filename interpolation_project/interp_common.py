import tensorflow as tf
import numpy as np
from common.model_components import build_mlp
from common.utility_fns import build_train_graph, get_loss


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
        loss = get_loss(mlp_output, y_) 
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
