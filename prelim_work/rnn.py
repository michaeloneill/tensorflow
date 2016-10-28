import tensorflow as tf
import numpy as np
import pdb

from utility_fns import train, build_train_graph, get_loss


def hidden_to_output(hiddens, dimHidden, nObs):

    """
    hiddens s * [bxh]
    output (s*b) x nObs

    """

    stdev = 1.0/np.sqrt(nObs)
    W_out = tf.Variable(tf.random_uniform([dimHidden, nObs], -stdev, stdev))

    packed_hiddens = tf.concat(0, hiddens)
    output = tf.matmul(packed_hiddens, W_out)
    return output


def build_rnn(params):

    with tf.device(params['device']):

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, params['seq_len'], params['nObs']], name='x')
            y_ = tf.placeholder(tf.float32, shape=[None, params['seq_len'], params['nObs']], name='y_')

        with tf.name_scope('lstm'):
            with tf.name_scope('lstm_cell'):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['dimHidden'], forget_bias=1.0, state_is_tuple=False)
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
            outputs = hidden_to_output(hiddens, params['dimHidden'], params['nObs']) # (s*b) x nObs


        with tf.name_scope('loss'):
            labels_reshape = tf.reshape(tf.transpose(y_, perm=[1,0,2]), [-1, params['nObs']]) # (s*b) x nObs
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

    params = {
        'dimHidden': 128,
        'num_layers': 1,
        'nObs': 7,
        'seq_len': 8,
        'dropout_keep_prob': 0.5,
        'miniBatchSize': 9,
        'epochs': 100,
        'learning_rate': 0.001,
        'device': '/cpu:0',
        'monitor_frequency': 10,
        'results_dir': '../results/rnn/logs/train'
    }


    data_filename='../data/rnn_trial_dataset.npz'

    data = np.load(data_filename)

    observations = data['observations']
    targets = data['targets']

    assert observations.shape[0]==targets.shape[0]

    model = build_rnn(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train(observations, targets, params, model, sess)


if __name__=='__main__':
    main()

        




    
                
            
                                

