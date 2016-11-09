
import tensorflow as tf

class BNBasicLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''

        def __init__(self, num_units, is_training, forget_bias=1.0):
                    self._num_units = num_units
                    self._is_training = is_training,
                    self._forget_bias = forget_bias # to reduce scale of forgetting at beginning of training. After this it will be learned as part of batch normalisation


         @property
         def state_size(self):
             return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

         @property
         def output_size(self):
             return self._num_units

         def __call__(self, x, state, scope=None):

            with tf.variable_scope(scope or type(self).__name__):
                c, h = state
                
                # change bias argument to False since batch_norm will add bias via shift
                # but forget_bias set manually to reduce forgetting at start of training when shift hasn't been learned
                # input-hidden and hidden-hidden weights built as one variable, to be split into portions for each gate
                concat = tf.nn.rnn_cell._linear([inputs, h], 4*self._num_units, False)

                i, j, f, o = tf.split(1, 4, concat) 

                # add batch_norm to each gate

                i = batch_norm(i, name_scope='i/', self._is_training)
                j = batch_norm(j, name_scope='j/', self._is_training)
                f = batch_norm(f, name_scope='f/', self._is_training)
                o = batch_norm(o, name_scope='o/', self._is_training)


                new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
                new_h = tf.tanh(batch_norm(new_c, 'new_h/', self._is_training)) * tf.sigmoid(o)

                new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)


                return new_h, new_state




def batch_norm(x, name_scope, is_training, epsilon=1e-3, decay=0.999):

    '''Assume 2d [batch, values] tensor. Based on common/layers/batch_norm_wrapper
    but now using variable scopes to allow variable sharing between cells'''
        

    with tf.variable_scope(name_scope):

        size = x.get_shape().as_list()[1]
        scale = tf.get_variable('scale', [size], initializer = tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', [size], initializer = tf.constant_initializer(0.0))

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean_op, train_var_op]):
            train_time = tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        test_time = tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return is_training*train_time + (1-is_training)*test_time


    # alternative for boolean is_training
    # def batch_statistics():
    #     with tf.control_dependencies([train_mean_op, train_var_op]):
    #         return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)


    # def population_statistics():

    #     return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

    # return tf.cond(training, batch_statistics, population_statistics)




    # # https://github.com/OlavHN/bnlstm/blob/master/lstm.py
    # # describes what _linear does
    # # see webpage for undeclared methods source code
    # def __call__(self, x, state, scope=None):

    #     with tf.variable_scope(scope or type(self).__name__):
    #         c, h = state

    #         x_size = x.get_shape().as_list()[1]

              # input-hidden and hidden-hidden weights built as one variable and then split into portions for each gate
    #         W_xh = tf.get_variable('W_xh',
    #                                [x_size, 4 * self._num_units],
    #                                initializer=orthogonal_initializer())
    #         W_hh = tf.get_variable('W_hh',
    #                                [self._num_units, 4 * self._num_units],
    #                                initializer=bn_lstm_identity_initializer(0.95))
    #         bias = tf.get_variable('bias', [4*self._num_units])

    #         xh = tf.matmul(x, W_xh)
    #         hh = tf.matmul(h, W_hh)

    #         bn_xh = batch_norm(xh, 'xh', self._is_training)
    #         bn_hh = batch_norm(hh, 'hh', self._is_training)

    #         hidden = bn_xh + bn_hh + bias


    #         i, j, f, o = tf.split(1, 4, hidden)

    #         new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
    #         bn_new_c = batch_norm(new_c, 'c', self._is_training)

    #         new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

    #         return new_h, (new_c, new_h)
