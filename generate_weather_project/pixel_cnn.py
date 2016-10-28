import tensorflow as tf
import numpy as np

from common.model_components import build_cnn_model
from common.utility_fns import train, get_wrapped_test_time_patches, mask_input
from pixel_mlp import generate_images, get_preds
import os
import pdb
from common.plotting import tile_raster_images
from PIL import Image

import h5py


def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    params_cnn = {
        'kernel_size': [[3,3], [3,3]],
        'num_outputs': [6, 12],
        'activations': ['relu', 'relu'],
        'fc_params':{'num_outputs': [512], 'activations': ['identity'], 'dropout':[False]}
    }

    params_train = {
        'miniBatchSize': 20,
        'epochs': 100,
        'learning_rate':0.001,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5,
        'loss_fn': 'softmax_cross_entropy_with_logits'
    }

    params = {
        'cnn': params_cnn,
        'train': params_train,
        'inpt_shape': {'x': [None, 10, 10, 2], 'y_': [None, 512]},
        'device':'/gpu:1',
        'results_dir': results_dir
    }

    
    # Load the training dataset

    ################## OVERFIT ######################
    training_data_filename = '../../data/generate_weather_project/wind/wind_201401_train_time_dataset_pixel_cnn_overfit.npz'
    #################################################

    
    # training_data_filename = '../../data/generate_weather_project/mnist_training_dataset_pixel_cnn_9_5.npz'
    # training_data_filename = '../../data/generate_weather_project/wind/wind_201401_train_time_dataset_pixel_cnn.npz'

    training_data = np.load(training_data_filename)

    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]

    model = build_cnn_model(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(train_set, val_set, test_set, params['train'], model, sess, results_dir)

    # load the testing dataset

    ################ OVERFIT ##########################
    i_filename = '../../data/generate_weather_project/wind/raw/wind_201401.h5'
    data = h5py.File(i_filename, 'r')['Dataset1']
    X_test_time = np.transpose(data[0][None, :], [0, 2, 3, 1]) # [N, H, W, n_channels]
    ###################################################


    # # testing_data_filename = '../../data/generate_weather_project/mnist_test_time_dataset.npz'
    # testing_data_filename = '../../data/generate_weather_project/wind/wind_201401_test_time_dataset.npz'
    
    # testing_data = np.load(testing_data_filename)
    # X_test_time = testing_data['X_test_time']

    
    p_i, p_j = 9, 5 # coordintates of pixel to predict in patch
    p_dim = 10
    tile_shape = (1,1)

    generate_images(X_test_time[:, :28, :28, :], params['train'], model,
                    sess, p_i, p_j, p_dim, params['results_dir'],
                    unroll=False, tile_shape=tile_shape)

    

if __name__=='__main__':
    main()
