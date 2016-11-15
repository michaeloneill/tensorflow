import tensorflow as tf
import numpy as np
import pdb

from common.utility_fns import train
from pixel_rnn import build_pixel_rnn_model
from pixel_mlp import generate_images, plot_channel_softmaxes_vs_ground_truth

import os



def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    params_rnn = {
        'cell_type': 'BNConvLSTMCell',
        'shape': (10, 10),
        'filter_size': (3,3),
        'num_output_feature_maps': 3,
        'num_layers': 1,
        'seq_len': 3,
        'out_activation': tf.identity,
        'dropout': False
    }
    
    params_train = {
        'miniBatchSize': 20,
        'epochs': 100,
        'learning_rate': 0.1,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'rnn': params_rnn,
        'train': params_train,
        'inpt_shape': {'x': [None, 3, 10, 10, 2], 'y_': [None, 512]},
        'channels_to_predict': [4,5],
        'device': '/gpu:1',
        'results_dir': results_dir
    }



    training_data_filename = '../../data/generate_weather_project/wind/historical/wind_201401_dataset_pixel_crnn_deltas/train_time.npz'    
    training_data = np.load(training_data_filename)
    
    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]
                                

    model = build_pixel_rnn_model(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(train_set, val_set, test_set, params['train'], model, sess, results_dir)

    # investigate softmax vs ground truth for selected examples that were trained on
    n = 3
    n_bins = params['inpt_shape']['y_'][1]/len(params['channels_to_predict'])
    ground_truth = np.reshape(train_set[1], [-1, n_bins, len(params['channels_to_predict'])])
    
    plot_channel_softmaxes_vs_ground_truth(train_set[0][:n], ground_truth[:n, :, :],
                                           model, sess, params['results_dir'])

    # load the testing dataset

    testing_data_filename = '../../data/generate_weather_project/wind/historical/wind_201401_dataset_pixel_crnn_deltas/test_time.npz'
    
    testing_data = np.load(testing_data_filename)
    X_test_time = testing_data['X_test_time']
    
    p_i, p_j = 9, 5 # coordintates of pixel to predict in patch
    p_dim = 10
    tile_shape = (2, 2) # for plotting results

    generate_images(X_test_time, params, model,
                    sess, p_i, p_j, p_dim,
                    tile_shape=tile_shape,
                    usage='crnn')

                                

if __name__=='__main__':
    main()

        




    
                
            
                                

