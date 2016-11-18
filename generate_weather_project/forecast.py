import tensorflow as tf
import numpy as np

from common.utility_fns import get_wrapped_test_time_patches
import os
import pdb
from common.plotting import tile_raster_images, concatenate_vert
from PIL import Image



def forecast(X_test_time, model, sess, params, usage='rnn'):

    '''
    X_test_time should be [N x H x W x depth] where depth is n_time_setps*num_components
    Samples should be in chronological order (as well as time steps within a sample)
    '''

    N, H, W, depth = X_test_time.shape

    n_components = params['n_components']
    n_time_steps = depth/n_components
    seq_len = params['seq_len']
    p_i = params['p_i']
    p_j = params['p_j']
    p_dim = params['p_dim']
    prediction_channels = np.arange(n_components*(seq_len-1), n_components*seq_len)
    n_forecasts = n_time_steps - seq_len + 1
    
    ground_truth = np.copy(X_test_time.reshape(N, H, W, n_time_steps, n_components))
    
    for k in xrange(n_forecasts):

        depth_start = k*n_components
        depth_end = depth_start + n_components*seq_len
        X_test_time[:, :, :, depth_start + prediction_channels] = 0
        
        for i in xrange(H):
            for j in xrange(W):
                patches = get_wrapped_test_time_patches(X_test_time[:, :, :, depth_start:depth_end], i, j, p_i, p_j, H, W, p_dim)
                patches = patches.reshape(-1, p_dim, p_dim, seq_len, n_components)
                patches = np.transpose(patches, [0, 3, 1, 2, 4])
                if usage is 'rnn':
                    patches = patches.reshape(-1, seq_len, p_dim*p_dim*n_components)

                # reuse storage      
                X_test_time[:, i, j, depth_start + prediction_channels] = sess.run(model['preds'],
                        feed_dict = {model['x']: patches,
                                     model['dropout_keep_prob']:1.0,
                                     model['is_training']:0.0
                        }
                )
            print 'generated row {} forecast {}/{}'.format(i, k+1, n_forecasts)

    X_test_time = X_test_time.reshape(N, H, W, n_time_steps, n_components)


    offset = (seq_len - 1) # start at first prediction
    for i in xrange(N):
        for j in xrange(offset, n_time_steps):

            im_forecast = Image.fromarray(tile_raster_images(
                X=np.transpose(X_test_time[i, :, :, j, :], [2, 0, 1]).reshape(-1, H*W),
                img_shape=(H, W),
                tile_shape=(1,n_components),
                tile_spacing=(1,1),
                scale_to_unit_interval=False))

            im_ground = Image.fromarray(tile_raster_images(
                X=np.transpose(ground_truth[i, :, :, j, :], [2, 0, 1]).reshape(-1, H*W),
                img_shape=(H, W),
                tile_shape=(1,n_components),
                tile_spacing=(1,1),
                scale_to_unit_interval=False))

            concat = concatenate_vert([im_forecast, im_ground])
            concat.save(params['results_dir'] + 'sample_{}_step_{}.png'.format(i, j))



def main():

    model_dir = input('Enter model file directory: ')
    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    params = {
        'n_components': 3,
        'seq_len': 3,
        'p_i': 9,
        'p_j': 5,
        'p_dim': 10,
        'results_dir': results_dir
    }
    
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_dir+'.meta')
    saver.restore(sess, model_dir)

    model = {
        'preds': tf.get_collection('preds')[0],
        'x': tf.get_collection('x')[0],
        'dropout_keep_prob': tf.get_collection('dropout_keep_prob')[0],
        'is_training': tf.get_collection('is_training')[0]
    }


    # load the testing dataset

    testing_data_filename = '../../data/generate_weather_project/wind/historical/wind_dataset_all_months/pixel_rnn_deltas/xlylp_forecasting/test_time.npz'
    
    testing_data = np.load(testing_data_filename)
    X_test_time = testing_data['X_test_time']


    forecast(X_test_time, model, sess, params, usage='rnn')
        


if __name__=='__main__':
    main()









    


    
    
