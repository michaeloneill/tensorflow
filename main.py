import numpy as np
import tensorflow as tf
from models import build_mlp_model, build_cnn_model, build_rnn_model
from training import train
import os


def main(USAGE='mlp'):
    '''Example usage of library '''
    
    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if USAGE is 'mlp':
        params_model = {
            'num_outputs': [100, 100, 512],
            'activations': ['relu', 'relu', 'identity'],
            'dropout': [False, False, True]
        }

    elif USAGE is 'rnn':
        params_model = {
            'kernel_size': [[3, 3], [3, 3]],
            'num_outputs': [6, 12],
            'activations': ['relu', 'relu'],
            'pool': [True, True],
            'fc_params': {'num_outputs': [512],
                          'activations': ['identity'],
                          'dropout': [False]
                          }
        }

    elif USAGE is 'rnn':
        params_model = {
            'cell_type': 'BasicLSTM',
            'dim_hidden': 100,
            'num_layers': 1,
            'seq_len': 7,
            'out_activation': tf.identity,
            'dropout': False
        }

    else:
        raise ValueError('usage not recognised')

    params_train = {
        'miniBatchSize': 20,
        'epochs': 10,
        'learning_rate': 0.01,
        'dropout_keep_prob': 0.5,  # if dropout layers exist
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'model': params_model,
        'train': params_train,
        'inpt_shape': {'x': [None, 800], 'y_': [None, 512]},
        'device': '/gpu:1',
        'results_dir': results_dir
    }

    if USAGE is 'mlp':
        model = build_mlp_model(params)
    elif USAGE is 'cnn':
        model = build_cnn_model(params)
    elif USAGE is 'rnn':
        model = build_rnn_model(params)
    else:
        raise ValueError('Usage not recognised')
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Load the training dataset
    training_data = np.load('my_training_set.npz')

    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]

    train(train_set, val_set,
          test_set, params['train'],
          model, sess, results_dir)

if __name__ == '__main__':
    main()
