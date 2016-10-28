import tensorflow as tf
import numpy as np

from common.model_components import build_mlp, build_train_graph
from common.utility_fns import train, get_wrapped_test_time_patches, mask_input
import os
import pdb
from common.plotting import tile_raster_images, images_to_tuple
from PIL import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py


def plot_channel_softmaxes_vs_ground_truth(inpt, ground_truth, model, sess, results_dir):

    ''' ground_truth must be [N, n_bins, n_channels] '''

    n_bins = ground_truth.shape[1]
    channel_softmaxes = sess.run(model['channel_softmaxes'],
                                 feed_dict = {model['x']: inpt,
                                              model['dropout_keep_prob']:0.0,
                                              model['is_training']: 0.0}
                                 )

    n_rows = inpt.shape[0]
    n_cols = len(channel_softmaxes)
    
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
    index = np.arange(n_bins)
    bar_width = 0.35

    for i, chan_axes in enumerate(axes):
        for j, ax in enumerate(chan_axes):
            ax.bar(index, channel_softmaxes[j][i], bar_width, color='b')
            ax.bar(index+bar_width, ground_truth[i, :, j], bar_width, color='r')

            ax.set_xticks([])
            # ax.set_xticks(index+bar_width)
            # ax.set_xticklabels(index)
    
            ax.set_ylabel('probability')
            ax.set_yticks([])
    fig.savefig(results_dir+'softmax_outputs_vs_ground_truth.png')

    

def get_channel_softmaxes(logits, n_channels):

    n_bins = logits.get_shape()[1].value/n_channels
    logits = tf.reshape(logits, [-1, n_bins, n_channels])

    channel_softmaxes = []
    for i in range(n_channels):
        with tf.name_scope('softmax_channel_{}'.format(i)):
            softmax = tf.nn.softmax(logits[:, :, i])
        channel_softmaxes.append(softmax)

    return channel_softmaxes


def get_channel_loss(channel_logits, channel_targets):

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(channel_logits, channel_targets))
    

def get_total_loss(logits, y_, n_channels):

    n_bins = logits.get_shape()[1].value/n_channels
    logits = tf.reshape(logits, [-1, n_bins, n_channels])
    targets = tf.reshape(y_, [-1, n_bins, n_channels])

    channel_losses = []
    for i in range(n_channels):
        with tf.name_scope('loss_channel_{}'.format(i)):
            loss = get_channel_loss(logits[:, :, i], targets[:, :, i])
        tf.scalar_summary('loss_channel_{}'.format(i), loss)
        channel_losses.append(loss)
            
    return tf.reduce_sum(channel_losses)


def build_pixel_mlp_model(params):

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
        is_training = tf.placeholder(tf.float32, name='is_training')
            
    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, is_training, params['mlp'])

    # for monitoring
    with tf.name_scope('channel_softmaxes'):
        channel_softmaxes = get_channel_softmaxes(mlp_output, params['n_channels'])
        
    with tf.name_scope('loss_total'):
        loss = get_total_loss(mlp_output, y_, params['n_channels'])
    tf.scalar_summary('loss_total', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': mlp_output,
        'channel_softmaxes': channel_softmaxes
    }
    
    return model




def get_preds(output, n_channels):

    ''' output is [N x n_bins*n_channels]
    Each row is arranged [bin_0_R, bin_0_G, bin_0_B, bin_1_R, bin_1_G, bin_1_B,.....bin_256_R, bin_256_G, bin_256_B]
    '''
    n_bins = output.get_shape()[1].value/n_channels
    with tf.name_scope('preds'):
        output = tf.reshape(output, [-1, n_bins, n_channels])
        preds = tf.argmax(output, 1) # [N x n_channels]
        # convert back to [0, 1)
        preds = tf.cast(preds, tf.float32)/255 # to be more precise reverse the digitize
    return preds



def generate_images(X_test_time, params, model, sess, p_i, p_j, p_dim, results_dir, unroll, tile_shape):

    ''' 
    X_test_time is [N x H x W x n_channels]
    X_test_time[:, H/2:, :, :] has been zeroed
    and goal is to regenerate it

    '''
    n = X_test_time.shape[0]
    H = X_test_time.shape[1]
    W = X_test_time.shape[2]
    n_channels = X_test_time.shape[3]

    for i in range(n_channels):    
        ground_truth_images = Image.fromarray(tile_raster_images(
            X=X_test_time[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        ground_truth_images.save(results_dir + 'ground_truth_images_channel_{}.png'.format(i))
    
    for i in range(H/2, H):
        for j in range(W):
            patches = get_wrapped_test_time_patches(X_test_time, i, j, p_i, p_j, H, W, p_dim)
            patches = mask_input(patches, p_i, p_j)
            if unroll: # mlp use
                patches = patches.reshape(-1, p_dim*p_dim*n_channels)
            
            X_test_time[:, i, j, :] = sess.run(get_preds(model['logits'], n_channels),
                    feed_dict = {model['x']: patches,
                                 model['dropout_keep_prob']:0.0,
                                 model['is_training']:0.0
                    }
            )

        print 'generated row {}'.format(i)

    for i in range(n_channels):    
        generated_images = Image.fromarray(tile_raster_images(
            X=X_test_time[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        generated_images.save(results_dir + 'generated_images_channel_{}.png'.format(i))


def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    params_mlp = {
        'num_outputs': [100, 100, 512],
        'activations': ['relu', 'relu', 'identity'],
        'dropout': [False, False, False]
    }


    params_train = {
        'miniBatchSize': 20,
        'epochs':2,
        'learning_rate':0.01,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'mlp': params_mlp,
        'train': params_train,
        'inpt_shape': {'x': [None, 200], 'y_': [None, 512]},
        'n_channels': 2,
        'device':'/gpu:1',
        'results_dir': results_dir
    }

    
    # Load the training dataset

    # training_data_filename = '../../data/generate_weather_project/mnist_training_dataset_pixel_mlp_9_5.npz'
    # training_data_filename = '../../data/generate_weather_project/wind/wind_201401_train_time_dataset_pixel_mlp.npz'

    ################## IMAGE OVERFIT ######################
    training_data_filename = '../../data/generate_weather_project/wind/wind_201401_train_time_dataset_pixel_mlp_overfit.npz'
    #################################################
    
    training_data = np.load(training_data_filename)

    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]    

    model = build_pixel_mlp_model(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(train_set, val_set, test_set, params['train'], model, sess, results_dir)

    
    # investigate softmax vs ground truth for selected examples
    n = 2
    n_bins = params['inpt_shape']['y_'][1]/params['n_channels']
    ground_truth = np.reshape(train_set[1], [-1, n_bins, params['n_channels']])
    plot_channel_softmaxes_vs_ground_truth(train_set[0][:n], ground_truth[:n, :, :],
                                           model, sess, params['results_dir'])

    # load the testing dataset

    # testing_data_filename = '../../data/generate_weather_project/mnist_test_time_dataset.npz'
    # testing_data_filename = '../../data/generate_weather_project/wind/wind_201401_test_time_dataset.npz'
    # testing_data = np.load(testing_data_filename)
    # X_test_time = testing_data['X_test_time']

    ################ IMAGE OVERFIT ##########################
    i_filename = '../../data/generate_weather_project/wind/raw/wind_201401.h5'
    data = h5py.File(i_filename, 'r')['Dataset1']
    X_test_time = np.transpose(data[0][None, :], [0, 2, 3, 1]) # [N, H, W, n_channels]
    ###################################################
    
    p_i, p_j = 9, 5 # coordintates of pixel to predict in patch
    p_dim = 10
    tile_shape = (1,1) # for plotting results

    generate_images(X_test_time[:, :28, :28, :], params['train'], model,
                    sess, p_i, p_j, p_dim, params['results_dir'],
                    unroll=True, tile_shape=tile_shape)

    

if __name__=='__main__':
    main()
