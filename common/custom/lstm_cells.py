import tensorflow as tf
import pdb

class BNLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025
    modelled on BasicLSTMCell https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py'''

    def __init__(self, num_units, is_training, forget_bias=1.0):
        self._num_units = num_units
        self._is_training = is_training
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
            concat = tf.nn.rnn_cell._linear([x, h], 4*self._num_units, bias=False)

            i, j, f, o = tf.split(1, 4, concat) 

            # add batch_norm to each gate

            i = batch_norm(i, 'i/', self._is_training)
            j = batch_norm(j, 'j/', self._is_training)
            f = batch_norm(f, 'f/', self._is_training)
            o = batch_norm(o, 'o/', self._is_training)


            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(batch_norm(new_c, 'new_h/', self._is_training)) * tf.sigmoid(o)

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


class ConvRNNCell(object):

    """Abstract object representing analagous to RNNCell but for a Convolutional RNN cell.
    adapted from https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow/blob/master/BasicConvLSTMCell.py
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, b_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          b_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[b_size x shape[0] x shape[1] x num_output_feature_maps]
          filled with zeros
        """
        shape = self.shape
        num_output_feature_maps = self.num_output_feature_maps
        zeros = (tf.zeros([b_size, shape[0], shape[1], num_output_feature_maps]),
                 tf.zeros([b_size, shape[0], shape[1], num_output_feature_maps]))

        return zeros


class BNConvLSTMCell(ConvRNNCell):

    """Basic Conv LSTM recurrent network cell based on tensorflow BasicLSTM cell
    with batch normalisation as described in arxiv.org/abs/1603.09025
    adapted from https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow/blob/master/BasicConvLSTMCell.py
    """

    def __init__(self, shape, filter_size, num_output_feature_maps, is_training, forget_bias=1.0):
        """Initialize the basic Conv LSTM cell.
        Args:
        shape: int tuple that is height and width of cell
        filter_size: int tuple thats the height and width of the filter
        num_output_feature_maps: int thats the depth of the cell 
        forget_bias: float, The bias added to forget gates (see above).
        """
        self.shape = shape # (H, W)
        self.filter_size = filter_size
        self.num_output_feature_maps = num_output_feature_maps
        self._is_training = is_training
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple((self.shape[0], self.shape[1], self.num_output_feature_maps),
                                             (self.shape[0], self.shape[1], self.num_output_feature_maps))
    @property
    def output_size(self):
        return (self.shape[0], self.shape[1], self.num_output_feature_maps)

    def __call__(self, x, state, scope=None):

        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            
            c, h = state

            concat = _conv_linear([x, h], self.filter_size, self.num_output_feature_maps * 4)
            i, j, f, o = tf.split(3, 4, concat) # split along output_feature_maps

            # # add batch_norm to each gate

            # i = batch_norm(i, 'i/', self._is_training, conv = True)
            # j = batch_norm(j, 'j/', self._is_training, conv = True)
            # f = batch_norm(f, 'f/', self._is_training, conv = True)
            # o = batch_norm(o, 'o/', self._is_training, conv = True)
                
            new_c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * tf.tanh(j)
            # new_h = tf.tanh(batch_norm(new_c, 'new_h/', self._is_training, conv=True)) * tf.sigmoid(o)
            new_h = tf.tanh(new_c)* tf.sigmoid(o)
                
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def _conv_linear(args, filter_size, num_output_feature_maps, scope=None):

    """convolution:
    Args:
      args: a list of 4D Tensors b x H x W x num_input_feature_maps (e.g. [x,h])
      filter_size: int tuple of filter height and width.
      num_output_feature_maps: int, number of output feature maps.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [b, h, w, num_output_feature_maps]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    Note: no biases - accounted for by offset in batch_norm
    """

    total_input_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[3] of arguments: %s" % str(shapes))
        else:
            total_input_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        filtr = tf.get_variable(
            "filter",
            [filter_size[0], filter_size[1], total_input_depth, num_output_feature_maps],
            initializer = tf.truncated_normal_initializer(),
            dtype=dtype
        )
        if len(args) == 1:
            return tf.nn.conv2d(args[0], filtr, strides=[1, 1, 1, 1], padding='SAME')
        else:
            # no need to apply conv2d to x and h seperately and sum because they have SAME dimensions and conv2d sums along 3rd axis
            return tf.nn.conv2d(tf.concat(3, args), filtr, strides=[1, 1, 1, 1], padding='SAME')
                                                                      

def batch_norm(x, name_scope, is_training, epsilon=1e-3, decay=0.999, conv=False):

    '''Assume 2d [batch, values] tensor. Based on common/layers/batch_norm_wrapper
    but now using variable scopes to allow variable sharing between cells'''
        

    with tf.variable_scope(name_scope):

        size = x.get_shape().as_list()[-1]
        scale = tf.get_variable('scale', [size], initializer = tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', [size], initializer = tf.constant_initializer(0.0))

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)

        if conv:
            batch_mean, batch_var = tf.nn.moments(x,[0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(x,[0])

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
