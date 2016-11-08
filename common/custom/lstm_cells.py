class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''

        def __init__(self, num_units, is_training):
                    self.num_units = num_units
                    self.is_training = is_training


         @property
         def state_size(self):
             return (self.num_units, self.num_units)

         @property
             def output_size(self):
                 return self.num_units

        def __call__(self, x, state, scope=None):

            with tf.variable_scope(scope or type(self).__name__):
                c, h = state
                x_size = x.get_shape().as_list()[1]
                W_xh = tf.get_variable('W_xh',
                                       [self.num_units, 4 * self.num_units],
                                       initializer=bn_lstm_identity_initializer(0.95))
                W_hh = tf.get_variable('W_hh',
                                       [self.num_units, 4 * self.num_units],
                                       initializer=bn_lstm_identity_initializer(0.95))
                bias = tf.get_variable('bias', [4*self.num_units])

                xh = tf.matmul(x, W_xh)
                hh = tf.matmul(h, W_hh)

                bn_xh = batch_norm(xh, 'xh', self.is_training)
                bn_hh = batch_norm(hh, 'hh', self.is_training)

                hidden = bn_xh + bn_hh + bias


                i, j, f, o = tf.split(1, 4, hidden)

                new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
                bn_new_c = batch_norm(new_c, 'c', self.is_training)

                new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

                return new_h, (new_c, new_h)

