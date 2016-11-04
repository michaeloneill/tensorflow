import tensorflow as tf
import numpy as np
import pdb

from utility_fns import train, build_train_graph, get_loss


def hidden_to_output(hiddens, dim_hidden, dim_output):

    """
    hiddens s * [bxh]
    output (s*b) x dim_output

    """

    stdev = 1.0/np.sqrt(dim_output)
    W_out = tf.Variable(tf.random_uniform([dim_hidden, dim_output], -stdev, stdev))

    packed_hiddens = tf.concat(0, hiddens)
    output = tf.matmul(packed_hiddens, W_out)
    return output


def build_rnn(params):

    with tf.device(params['device']):

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, params['seq_len'], params['dim_input']], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, params['seq_len'], params['dim_input']], name='y_')

        with tf.name_scope('lstm'):
            with tf.name_scope('lstm_cell'):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['dim_hidden'], forget_bias=1.0, state_is_tuple=False)
            with tf.name_scope('dropped_cell'):
                dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                dropped_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob)
            with tf.name_scope('stacked_lstm'):
                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropped_cell]*params['num_layers'], state_is_tuple=False)
                state = stacked_lstm.zero_state(params['miniBatchSize'], tf.float32)

        with tf.name_scope('hiddens'):
            hiddens = [None]*params['seq_len']
            for t in range(params['seq_len']):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                hiddens[t], state = lstm_cell(x[:, t, :], state) # hiddens is s * [b x h]
                               

        with tf.name_scope('output'):
            outputs = hidden_to_output(hiddens, params['dim_hidden'], params['dim_output']) # (s*b) x dim_output


        with tf.name_scope('loss'):
            labels_reshape = tf.reshape(tf.transpose(y_, perm=[1,0,2]), [-1, params['dim_output']]) # (s*b) x dim_output
            loss = get_loss(outputs, labels_reshape)
        tf.scalar_summary('loss', loss)
            
                
        model = {
            'x': x,
            'y_': y_,
            'dropout_keep_prob': dropout_keep_prob,
            'loss': loss,
            'train': build_train_graph(loss, params)
        }

        return model



def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    params_rnn = {
        'dim_input': 200,
        'dim_hidden': 100,
        'dim_output': 512,
        'num_layers': 1,
        'seq_len': 4
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
        'inpt_shape': {'x': [None, 4, 200], 'y': [None, 512]}
        'device': '/gpu:1',
        'results_dir': results_dir
    }




    training_data_filename = '../../data/generate_weather_project/wind/historical/256_bin/wind_201401_dataset_pixel_mlp_historical/train_time.npz'    
    training_data = np.load(data_filename)


    # do this here and to each patch during generate_images loop
    X_train = training_data['X_train'].reshape(-1, H, W, 4, 2)
    X_train = X_train.transpose(0, 3, 1, 2, 4)
    X_train = X_train.reshape(-1, 4, H*W*2)
    
    train_set = [training_data['X_train'], training_data['y_train']]
    val_set = [training_data['X_val'], training_data['y_val']]
    test_set = [training_data['X_test'], training_data['y_test']]    

    assert observations.shape[0]==targets.shape[0]

    model = build_rnn(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(observations, targets, params, model, sess)


if __name__=='__main__':
    main()

        




    
                
            
                                

