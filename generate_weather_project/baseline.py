import numpy as np
import pdb

from common.model_components import build_rnn
from common.utility_fns import train
from pixel_mlp import generate_images, plot_channel_softmaxes_vs_ground_truth
from pixel_rnn import hidden_to_output

import os

def get_preds(X_test_time, y_test_time, params, model, sess, tile_shape):

    ''' X_test_time is [N x seq_len x H x W x num_channels]
    '''

    n, s, H, W, n_channels = X_test_time.shape

    y_test_time = y_test_time.reshape(-1, H, W, n_channels)
    for i in range(n_channels):    
        ground_truth_images = Image.fromarray(tile_raster_images(
            X=y_test_time[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        ground_truth_images.save(params['results_dir'] + 'ground_truth_images_channel_{}.png'.format(i))


    output = sess.run(model['logits'],
                      feed_dict = {model['x']: X_test_time.reshape(-1, s, H*W*n_channels),
                                   model['dropout_keep_prob']: 1.0,
                                   model['is_training']: 0.0
                      }
    )
    
    output = output.reshape(-1, H, W, n_channels)
    for i in range(num_channels):
        generated_images = Image.fromarray(tile_raster_images(
            X=output[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        ground_truth_images.save(params['results_dir'] + 'generated_images_channel_{}.png'.format(i))



def get_preds(X_test_time, params, model, sess, tile_shape):

    ''' X_test_time is [N x H x W x seq_len*num_channels]
    '''

    n, H, W, depth = X_test_time.shape
    
    for i in params['channels_to_predict']:    
        ground_truth_images = Image.fromarray(tile_raster_images(
            X=X_test_time[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        ground_truth_images.save(params['results_dir'] + 'ground_truth_images_channel_{}.png'.format(i))

    # remove channels we want to predict
    X_test_time = np.delete(X_test_time, params['channels_to_predict'], 3)
    depth = X_test_time.shape[-1]

    # reshape to N x seq_len x H x W x num_channels
    num_channels = depth/params['rnn']['seq_len']
    X_test_time.reshape(-1, H, W, params['rnn']['seq_len'], num_channels)
    X_test_time.transpose(X_test_time, [0, 3, 1, 2, 4])
    X_test_time.reshape(-1, params['rnn']['seq_len'], H*W*num_channels)

    output = sess.run(model['logits'],
                      feed_dict = {model['x']: X_test_time,
                                   model['dropout_keep_prob']: 1.0,
                                   model['is_training']: 0.0
                      }
    )
    
    output = output.reshape(-1, H, W, num_channels)
    for i in range(num_channels):
        generated_images = Image.fromarray(tile_raster_images(
            X=output[:, :, :, i].reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=tile_shape,
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        ground_truth_images.save(params['results_dir'] + 'generated_images_channel_{}.png'.format(i))


def get_loss(logits, targets):

    return tf.reduce_mean(tf.reduce_sum(tf.square(logits-targets), reduction_indices=[1]))



def build_baseline_rnn_model(params):


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
        outputs = hidden_to_output(hiddens[-1], params['inpt_shape']['y_'][1]) # (s*b) x dim_output
        
    with tf.name_scope('loss'):
        loss = get_loss(outputs, y_)
    tf.scalar_summary('loss', loss)            
                
    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': outputs
    }

    return model






def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    params_rnn = {
        'cell_type': 'BasicLSTMCell',
        'dim_hidden': 181*360*2,
        'num_layers': 1,
        'seq_len': 2,
        'dropout': False
    }
    
    params_train = {
        'miniBatchSize': 20,
        'epochs': 10,
        'learning_rate': 0.01,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'rnn': params_rnn,
        'train': params_train,
        'inpt_shape': {'x': [None, 2, 181*360*2], 'y_': [None, 181*360*2]},
        'channels_to_predict': [4,5],
        'device': '/gpu:1',
        'results_dir': results_dir
    }


    training_data_filename = '../../data/generate_weather_project/wind/historical/wind_201401_dataset_pixel_rnn_deltas_baseline/train_time.npz'    
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

    testing_data_filename = '../../data/generate_weather_project/wind/historical/wind_201401_dataset_pixel_rnn_deltas/test_time.npz'
    
    testing_data = np.load(testing_data_filename)
    X_test_time = testing_data['X_test_time']
    
    tile_shape = (2, 2) # for plotting results

    get_preds(X_test_time, params, model, sess, tile_shape)

                                

if __name__=='__main__':
    main()
