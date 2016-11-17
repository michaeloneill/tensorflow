import tensorflow as tf
import numpy as np
import pdb

from common.model_components import build_rnn, build_train_graph
from common.utility_fns import train
from pixel_mlp import generate_images, get_preds, get_total_loss, get_channel_softmaxes, plot_channel_softmaxes_vs_ground_truth

import os


def hidden_to_output(hiddens, dim_output, activation = tf.identity):

    """
    
    hiddens can be s * [bxh] or [bxh] (fully connected cell layers)
    or it can be s*[bxHxWxnum_output_feature_maps] or [bxHxWxnum_output_feature_maps] (convolutional layers) 

    Either way
      output (s*b) x dim_output or b*dim_output

    """        

    if type(hiddens) is list:
        hiddens = tf.concat(0, hiddens)
    if len(hiddens.get_shape().as_list()) == 4: # convolutional cell
        dim_hidden = np.prod(hiddens.get_shape().as_list()[1:])
        hiddens = tf.reshape(hiddens, [-1, dim_hidden])
    else:
        assert len(hiddens.get_shape().as_list()) == 2
        dim_hidden = hiddens.get_shape()[-1].value

    stdev = 1.0/np.sqrt(dim_output)
    W_out = tf.Variable(tf.random_uniform([dim_hidden, dim_output], -stdev, stdev))

    return activation(tf.matmul(hiddens, W_out))
                                       


def build_pixel_rnn_model(params):


    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=params['inpt_shape']['x'], name='x')
        y_ = tf.placeholder(tf.float32, shape=params['inpt_shape']['y_'], name='y_')

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
    
    with tf.name_scope('lstm'):
        hiddens, _ = build_rnn(x, dropout_keep_prob, is_training, params['rnn'])
            
    with tf.name_scope('output'):
        outputs = hidden_to_output(hiddens[-1], params['inpt_shape']['y_'][1], params['rnn']['out_activation']) # (s*b) x dim_output

    with tf.name_scope('predictions'):
        preds = get_preds(outputs, len(params['channels_to_predict']))
        
    # for monitoring
    with tf.name_scope('channel_softmaxes'):
        channel_softmaxes = get_channel_softmaxes(outputs, len(params['channels_to_predict']))
        
    with tf.name_scope('loss_total'):
        loss = get_total_loss(outputs, y_, len(params['channels_to_predict']))
    tf.scalar_summary('loss_total', loss)            
                
    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': outputs,
        'preds': preds,
        'channel_softmaxes': channel_softmaxes
    }

    return model
                        


def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    params_rnn = {
        'cell_type': 'BasicLSTM',
        'dim_hidden': 100,
        'num_layers': 1,
        'seq_len': 3,
        'out_activation': tf.identity,
        'dropout': False
    }
    
    params_train = {
        'miniBatchSize': 20,
        'epochs': 10,
        'learning_rate': 0.1,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'rnn': params_rnn,
        'train': params_train,
        'inpt_shape': {'x': [None, 3, 300], 'y_': [None, 768]},
        'channels_to_predict': [6, 7, 8],
        'device': '/gpu:1',
        'results_dir': results_dir
    }



    training_data_filename = '../../data/generate_weather_project/wind/historical/wind_dataset_all_months/pixel_rnn_deltas/xlylp/train_time.npz'    
    training_data = np.load(training_data_filename)
    
    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]
                                

    model = build_pixel_rnn_model(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(train_set, val_set, test_set, params['train'], model, sess, results_dir) # graph restored with optimal variables

    # investigate softmax vs ground truth for selected examples that were trained on
    n = 3
    n_bins = params['inpt_shape']['y_'][1]/len(params['channels_to_predict'])
    ground_truth = np.reshape(train_set[1], [-1, n_bins, len(params['channels_to_predict'])])
    
    plot_channel_softmaxes_vs_ground_truth(train_set[0][:n], ground_truth[:n, :, :],
                                           model, sess, params['results_dir'])

    # load the testing dataset

    testing_data_filename = '../../data/generate_weather_project/wind/historical/wind_dataset_all_months/pixel_rnn_deltas/xlylp/test_time.npz'
    
    testing_data = np.load(testing_data_filename)
    X_test_time = testing_data['X_test_time']
    
    p_i, p_j = 9, 5 # coordintates of pixel to predict in patch
    p_dim = 10
    tile_shape = (4, 4) # for plotting results

    generate_images(X_test_time, params, model,
                    sess, p_i, p_j, p_dim,
                    tile_shape=tile_shape,
                    usage='rnn')

                                

if __name__=='__main__':
    main()

        




    
                
            
                                

