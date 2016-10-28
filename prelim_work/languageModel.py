import tensorflow as tf
import numpy as np
import pdb
import random
import tokenizer


random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

PAD_INDEX = 0
START_INDEX = 2
END_INDEX = 3

"""
Dimensions:
b = batch size (None)
s = sequence length
e = embedding size
h = hidden size
v = vocab size
"""


def embed_seq(seq, voc_size, embedding_size):
    """Embed a seq from a voc_size dictionary to embedding_size vectors

    Input:
        seq: [b x s] tensor of dictionary ids

    Output:
        [b x s x e] tensor of embeddings
    """
    stdev = 1.0/np.sqrt(embedding_size)
    embedding = tf.Variable(tf.random_uniform([voc_size, embedding_size], -stdev, stdev))
    return tf.nn.embedding_lookup(embedding, seq)

def hidden_to_output(hiddens, last_hidden_size, voc_size):
    """Convert last hidden layer to a distribution over the vocab

    Input:
        hiddens: list of s * [b x h]

    Output:
        a [(s*b) x v] tensor where each [,:] is an unnormalized log distribution over the vocab
    """
    stdev = 1.0/np.sqrt(voc_size)
    output_matrix = tf.Variable(tf.random_uniform([last_hidden_size, voc_size], -stdev, stdev)) # [h x v]
    packed_hiddens = tf.concat(0, hiddens) # [(s*b) x h]
    return tf.matmul(packed_hiddens, output_matrix) # [(s*b) x v]
    

def get_loss(unnorm_logits, labels, padding_mask):
    """
    Input:
        unnorm_logits: [(s*b) x v] tensor of unnormalized log distributions over the vocab
        labels: [b x s] tensor of ground truth token ids
        padding_mask: [b x s] tensor 1 if is not padding else 0
    
    Output:
        the per-token average softmax cross entropy loss
    """
    all_scores = tf.nn.sparse_softmax_cross_entropy_with_logits(unnorm_logits, 
                    tf.reshape(tf.transpose(labels), [-1]) # (s*b)
                    ) # (s*b)
    unpadded_scores = tf.reshape(tf.transpose(padding_mask), [-1] ) * all_scores # (s*b)
    per_token_average_loss = tf.reduce_sum(unpadded_scores) / tf.reduce_sum(padding_mask)
    return per_token_average_loss


def build_LSTM_language_model(params):
    """Build a language model

    """
    with tf.device(params['device']):

        input_seq = tf.placeholder(tf.int32, [None, params['seq_len']]) # [b x s]
        gt_label = tf.placeholder(tf.int32, [None, params['seq_len']]) # [b x s]
        gt_mask = tf.placeholder(tf.float32, [None, params['seq_len']]) # [b x s]

        dropout_keep_prob = tf.placeholder(tf.float32)

        bat_size = tf.shape(input_seq)[0]

        # embed
        embeddings = embed_seq(input_seq, params['vocab_size'], params['embedding_size']) # [b x s x e]

        # lstm
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hid_size'], forget_bias=1.0, state_is_tuple=False)
        dropped_cell = tf.nn.rnn_cell.DoropoutWrapper(lstm_cell, input_keep_prob = dropout_keep_prob)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropped_cell] * params['num_layers'], state_is_tuple=False)
        state = stacked_lstm.zero_state(bat_size, tf.float32)

        hiddens = [None]*params['seq_len']
        for t in range(params['seq_len']):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            hiddens[t], state = lstm_cell(embeddings[:, t, :], state)

        # output
        logits = hidden_to_output(hiddens, params['hid_size'], params['vocab_size']) # [(s*b) x v]
        sm_logits = tf.nn.softmax(logits) # [(s*b) x v]
        #loss
        per_word_average_loss = get_loss(logits, gt_label, gt_mask)
        
        model = {
            'input_seq': input_seq,
            'gt_label': gt_label,
            'gt_mask': gt_mask,
            'dropout_keep_prob': dropout_keep_prob,
            'loss': per_word_average_loss, 
            'train': make_train_graph(per_word_average_loss, params),
            'softmax_output': sm_logits}

        return model


def make_train_graph(loss, params):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), params['grad_clip'])

    optimizer = tf.train.MomentumOptimizer(params['learning_rate'], momentum=params['momentum'])
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    return train_step

def training_time(data, params, model, sess):
    batches = make_batches(data, params['batch_size'])
    for iter,b in enumerate(batches):
        _, loss = sess.run([model['train'], model['loss']],
            { model['input_seq']: b['source'],
            model['gt_label']:  b['target'],
            model['gt_mask']:   b['mask'],
            model['dropout_keep_prob']: params['dropout_keep_prob']
            }
        )

        if (iter%10 == 0):
            print "train:", loss

def validation_time(data, params, model, sess):
    batches = make_batches(data, params['batch_size'])
    total_loss = 0
    for iter,b in enumerate(batches):
        loss = sess.run(model['loss'],
            { model['input_seq']: b['source'],
            model['gt_label']:  b['target'],
            model['gt_mask']:   b['mask'],
            model['dropout_keep_prob']: 1
            }
        )
        total_loss += loss
    
    print "valid:", total_loss/len(batches)

def sample_from_softmax(sm, temp):
    z = -np.random.gumbel(loc=0, scale=1, size=sm.shape)
    return (sm/temp + z).argmax()

def sample_generation(inv_vocab, params, model, sess):
    input = [PAD_INDEX] * params['seq_len']
    input[0] = START_INDEX

    i = 0
    while i < params['seq_len']-1 and input[i] is not END_INDEX:
        sm = sess.run(model['softmax_output'], { model['input_seq']: [input], model['dropout_keep_prob']: 1})
        input[i+1] = sample_from_softmax(sm[i], params['temperature'])
        i += 1
    
    print ' '.join(map(lambda x: inv_vocab[x], input))
    

def unit_test():
    params = {
        'vocab_size' : 10,
        'embedding_size': 3,
        'hid_size': 3,
        'device': '/cpu:0',
        'grad_clip': 1,
        'learning_rate': 0.1,
        'seq_len': 4
    }

    data = {}
    data['source'] = [[1,2,3,4]]
    data['target'] = [[2,3,4,5]]
    data['mask'] = [[1,1,1,1]]

    training_time(data, params)


def pad_data_instance(instance, seq_length):
    instance = instance[:seq_length+1] if len(instance) > seq_length+1 else instance    
    padding = [PAD_INDEX] * (seq_length + 1 - len(instance))
    flags = [1] * len(instance)
    
    padded_input = instance + padding
    source = padded_input[:seq_length]
    target = padded_input[1:]
    mask = (flags + padding)[1:] 

    return {'source': source, 'target': target, 'mask': mask}

def load_data(params, data_file):
    with open(data_file, 'r') as f:
        as_strings = f.readlines()
    data = map(lambda x : pad_data_instance(map(int, x.split(' ')), params['seq_len']), as_strings)
    return data    

def make_batches(data, batch_size):
    random.shuffle(data)
    n_batch = len(data) / batch_size # integer division
    batches = map(list, np.array(data)[:(n_batch*batch_size)].reshape(-1,batch_size))
    batched_data = []
    for i in range(len(batches)):
        batched_data.append({})
        for k in batches[0][0].keys():
            batched_data[i][k] = np.array([b[k] for b in batches[i]])

    return batched_data

def data_import_test():
    params = {}
    params['data_file'] = 'tokens_lstm-topcoder-test.txt'
    params['seq_len'] = 5

    data = load_data(params)

def get_vocab_size(vocab_file_name):
    with open(vocab_file_name, 'r') as f:
        voc_size = len(f.readlines())
    return voc_size

def main():
    vocab_size = get_vocab_size('vocab.txt')

    params = {
        'vocab_size' : vocab_size,
        'embedding_size': 128,
        'hid_size': 128,
        'device': '/gpu:1',
        'grad_clip': 5,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'seq_len': 512,
        'train_data_file': 'tokens_lstm-topcoder-train.txt',
        'valid_data_file': 'tokens_lstm-topcoder-val.txt',
        'batch_size': 10,
        'num_layers': 1,
        'dropout_keep_prob': 0.5,
        'temperature': 0.2
    }
    
    train_data = load_data(params,params['train_data_file'])
    valid_data = load_data(params,params['valid_data_file'])
    model = build_LSTM_language_model(params)
    inv_vocab = tokenizer.invert_vocab(tokenizer.load_vocab('vocab.txt'))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.initialize_all_variables())

    for epoch in range(999):
        training_time(train_data, params, model, sess)
        validation_time(valid_data, params, model, sess)
        if epoch%10 is 0:
            sample_generation(inv_vocab, params, model, sess)

    training_time(data, params)

if __name__ == '__main__':
    main()