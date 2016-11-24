import os
import tensorflow as tf
from models import build_mlp_model


def main():
    '''Example usage of library '''
    
    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Either
    params_mlp = {
        'num_outputs': [100, 100, 512],
        'activations': ['relu', 'relu', 'identity'],
        'dropout': [False, False, True]
    }
    
    # or
    params_rnn = {
        'cell_type': 'BasicLSTM',
        'dim_hidden': 100,
        'num_layers': 1,
        'seq_len': 7,
        'out_activation': tf.identity,
        'dropout': False
    }
    
    # or
    params_cnn = {
        'kernel_size': [[3,3], [3,3]],
        'num_outputs': [6, 12],
        'activations': ['relu', 'relu'],
        'pool': [True, True],
        'fc_params':{'num_outputs': [512],
                     'activations': ['identity'],
                     'dropout':[False]
    }

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
        'mlp': params_mlp, # change as required
        'train': params_train,
        'inpt_shape': {'x': [None, 800], 'y_': [None, 512]},
        'device': '/gpu:1',
        'results_dir': results_dir
    }

    model = build_mlp_model(params) # change as required
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

if __name__=='__main__':
    main()
