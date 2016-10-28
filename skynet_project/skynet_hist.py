import tensorflow as tf
import numpy as np
import pdb
from PIL import Image

from plotting import tile_raster_images, images_to_tuple
from utility_fns import train, unit_scale
from skynet_mlp import build_skynet_mlp, generate_predictions

import os


def main():


    data_filename='../data/hampstead_dataset.npz'

    results_dir = input('Enter results directory: ')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    data = np.load(data_filename)
    images = data['images'].astype(np.float32)
    observations = data['observations'].astype(np.float32)

    targets = np.array([
        np.concatenate(
            tuple(
                unit_scale(np.histogram(im[:, :, i],8)[0])
                for i in range(images.shape[-1])
            )
        )
        for im in images])

    print 'observations shape: {}'.format(observations.shape)
    print 'targets shape: {}'.format(targets.shape)
    print 'observations max/min: {}/{}'.format(observations.max(), observations.min())
    print 'targets max/min: {}/{}'.format(targets.max(), targets.min())

    params_mlp = {
        'inpt_dim': 11,
        'num_outputs': [100, 100, 24],
        'activations':['sigmoid', 'sigmoid', 'sigmoid'],
        'dropout': [False, False, False]
    }

    params_train = {
        'miniBatchSize': 20,
        'epochs': 100,
        'learning_rate':0.001,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'mlp': params_mlp,
        'train': params_train,
        'inpt_shape': [None, 11],
        'targets_shape': [None, 24],
        'device':'/gpu:1',
        'results_dir': results_dir
    }

    
    targets_hists_as_images = Image.fromarray(tile_raster_images(
        X=targets[:100],
        img_shape=[1, targets.shape[1]],
        tile_shape=(10,10),
        tile_spacing=(1,1),
        scale_to_unit_interval=False))
    targets_hists_as_images.save(params['results_dir']+'target_hists_as_images.png')

    # save images that target_channel_averages relate to
    target_images = Image.fromarray(tile_raster_images(
        X=images_to_tuple(images[:100]),
        img_shape=images.shape[1:3],
        tile_shape=(10,10),
        tile_spacing=(1,1),
        scale_to_unit_interval=False))
    target_images.save(params['results_dir']+'target_images.png')
    
    
    model = build_skynet_mlp(params)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
        
    train(observations, targets, params['train'], model, sess, params['results_dir'])

    generate_predictions(observations, [1, targets.shape[1]], params['train'],
                             model, sess, params['results_dir'])


if __name__=='__main__':
    main()
