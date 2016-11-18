import tensorflow as tf
import numpy as np

import os
import pdb
from common.plotting import tile_raster_images, images_to_tuple
from PIL import Image



def forecast(X, model, params, usage='rnn'):

    '''
    X should be [N x H x W x depth] where depth is n_time_setps*num_components
    Samples should be in chronological order (as well as time steps within a sample)
    '''

    N, H, W, depth = X.shape

    n_components = params['n_components']
    n_time_steps = depth/n_components
    seq_len = params['seq_len']
    p_i = params['p_i']
    p_j = params['p_j']
    p_dim = params['p_dim']
    prediction_channels = np.arange(n_components*(seq_len-1), n_components*seq_len)
    n_forecasts = n_time_steps - seq_len + 1
    
    ground_truth = X.reshape(N, H, W, n_time_steps, n_components)
    
    for k in xrange(n_forecasts):

        depth_start = k*n_components
        depth_end = depth_start + n_components*seq_len
        X[:, :, :, depth_start + prediction_channels] = 0
        
        for i in xrange(H):
            for j in xrange(W):
                patches = get_wrapped_test_time_patches(X[:, :, :, depth_start:depth_end], i, j, p_i, p_j, H, W, p_dim)
                patches = patches.reshape(-1, p_dim, p_dim, seq_len, n_components)
                patches = np.transpose(patches, [0, 3, 1, 2, 4])
                if usage is 'rnn':
                    patches = patches.reshape(-1, seq_len, p_dim*p_dim*n_components)

                # reuse storage      
                X[:, i, j, depth_start + prediction_channels] = sess.run(model['preds'],
                        feed_dict = {model['x']: patches,
                                     model['dropout_keep_prob']:1.0,
                                     model['is_training']:0.0
                        }
                )
            print 'generated row {} forecast {}/{}'.format(i, k, n_forecasts)

    X = X.reshape(N, H, W, n_time_steps, n_components)


    offset = (seq_len - 1) # start at first prediction
    for i in xrange(N):
        for j in xrange(offset, n_time_steps):

            im_forecast = Image.fromarray(tile_raster_images(
                X=np.transpose(X[i, :, :, j, :], [2, 0, 1]).reshape(-1, H*W),
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
            generated_images.save(params['results_dir'] + 'forecasts/sample_{}_step_{}.png'.format(i, j))

            











    


    
    
