import tensorflow as tf
import numpy as np
import pdb

from layers import convPoolLayer, fullyConnectedLayer, weight_variable, bias_variable, conv2d, max_pool_2x2
from utility_fns import train, build_train_graph, get_loss

    
def build_cnn(params):

    with tf.device(params['device']):
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32,
                               shape=[None,
                                      params['imHeight'],
                                      params['imWidth'],
                                      params['nChannels']],
                               name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, params['output_dim']],
                                name='y_')
        with tf.variable_scope('layer1') as scope:
            layer1 = convPoolLayer(x, params['filtr_dim'], params['nChannels'],
                                   params['nFeatureMaps'][0], 'layer1')
            scope.reuse_variables()
            
        with tf.variable_scope('layer2') as scope:
            layer2 = convPoolLayer(layer1, params['filtr_dim'], params['nFeatureMaps'][0],
                               params['nFeatureMaps'][1], 'layer2')
            scope.reuse_variables()
            layer2_flat = tf.reshape(layer2, [-1, params['imHeight']*params['imWidth']*params['nFeatureMaps'][1]/(2*2*2*2)])
            
        with tf.variable_scope('layer3') as scope:
            layer3 = fullyConnectedLayer(layer2_flat,
                                         params['nFeatureMaps'][1]*params['imWidth']*params['imHeight']/(2*2*2*2),
                                         params['dimHiddenFully'], 'layer3')
            scope.reuse_variables()

        with tf.name_scope('dropout'):
            dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            dropped = tf.nn.dropout(layer3, dropout_keep_prob)
        
        with tf.name_scope('output'):
            y = fullyConnectedLayer(dropped, params['dimHiddenFully'], params['output_dim'],
                                    'layer4', tf.identity)

        with tf.name_scope('loss'):
            loss = get_loss(y, y_)
        tf.scalar_summary('loss', loss)

        model = {
            'x': x,
            'y_': y_,
            'dropout_keep_prob': dropout_keep_prob,
            'loss': loss,
            'train': build_train_graph(loss, params)
            }

        return model
            

def main():

    params = {
        'imHeight': 60,
        'imWidth': 60,
        'nChannels': 3,
        'filtr_dim': 5,
        'nFeatureMaps': [6, 12],
        'dimHiddenFully': 100,
        'output_dim': 1,
        'miniBatchSize': 20,
        'dropout_keep_prob': 0.5,
        'epochs':2,
        'learning_rate':0.001,
        'device':'/cpu:0',
        'monitor_frequency': 10,
        'results_dir':'../results/cnn/logs/train'
    }

    
    data_filename='../data/images_targets.npz'

    data = np.load(data_filename)
    images = data['images']
    targets = data['targets']
    assert images.shape[0]==targets.shape[0]

    model = build_cnn(params)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(images, targets[:, None], params, model, sess)


if __name__== '__main__':

    main()
    
