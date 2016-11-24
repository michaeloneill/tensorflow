import tensorflow as tf
import numpy as np

LOSS_FNS = {
    'mean_squared': tf.square,
    'sigmoid_cross_entropy_with_logits': tf.nn.sigmoid_cross_entropy_with_logits,
    'softmax_cross_entropy_with_logits': tf.nn.softmax_cross_entropy_with_logits
}


def run_train(train_set, params, model, sess, miniBatchIndex, merged):
    
    return sess.run([model['train'], model['loss'], merged], feed_dict = {
        model['x']: train_set[0][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
        model['y_']: train_set[1][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
        model['dropout_keep_prob']:params['dropout_keep_prob'],
        model['is_training']:1.0
    }
    )


def run_val(val_set, params, model, sess, merged):

    return sess.run([model['loss'], merged], feed_dict = {
        model['x']: val_set[0],
        model['y_']: val_set[1],
        model['dropout_keep_prob']:1.0,
        model['is_training']:0.0
    }
    )

def run_test(test_set, params, model, sess):

    return sess.run(model['loss'], feed_dict = {
        model['x']: test_set[0],
        model['y_']: test_set[1],
        model['dropout_keep_prob']:1.0,
        model['is_training']:0.0
    }
    )

def train(train_set, val_set, test_set, params, model, sess, results_dir):
    
    saver = tf.train.Saver(max_to_keep=1)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(results_dir+'logs/train', sess.graph)
    val_writer = tf.train.SummaryWriter(results_dir+'logs/val', sess.graph)
    
    nBatchTrain = train_set[0].shape[0]/params['miniBatchSize']
    nBatchVal = val_set[0].shape[0]/params['miniBatchSize']
    nBatchTest = val_set[0].shape[0]/params['miniBatchSize']

    best_val_loss = np.inf
    best_model_file = None
    best_iter = None
    
    print 'starting training...'
    for epoch in range(params['epochs']):
        for miniBatchIndex in range(nBatchTrain):
            iteration = epoch*nBatchTrain + miniBatchIndex

            _, loss_train, summary = run_train(train_set, params, model, sess, miniBatchIndex, merged)
            train_writer.add_summary(summary, iteration)
            
            if (iteration+1)%params['monitor_frequency'] == 0:

                print 'train loss for minibatch {0}/{1} epoch {2} is: {3:.2f}'.format(
                    miniBatchIndex+1, nBatchTrain, epoch+1, loss_train)
                
                loss_val, summary = run_val(val_set, params, model, sess, merged) 
                val_writer.add_summary(summary, iteration) 
                print 'correpsonding val loss is: {:.2f}'.format(loss_val)

                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_iter = iteration
                    best_model_file = saver.save(sess, results_dir+'best_model', global_step=best_iter)


    saver.restore(sess, best_model_file) # now graph in session has optimal variables
    loss_test = run_test(val_set, params, model, sess) 
    print 'Training complete. Test set loss at lowest validation loss is: {:.2f}'.format(loss_test)



def mask_input(patches, row, col, channels):

    '''
    Zeros out input pixels including, below and to right of (row, col) to mimic an automaton.

    (row, col) represents coordinates of pixel we want to predict.
    patches is [n, patch_dim, patch_dim, C]
    channels is list of channels (patches.shape[-1]) that are to be masked

    Returns rastered masked patches [n, patch_dim, patch_dim, C]
    
    '''
    assert row < patches.shape[1] and col < patches.shape[2]
    
    for i in xrange(patches.shape[0]):
        patches[i, row+1:, :, channels] = 0
        patches[i, row, col:, channels] = 0

    return patches


def get_wrapped_test_time_patches(images, im_i, im_j, p_i, p_j, im_H, im_W, p_dim):

    ''' 
    Returns patch on each image in images where (p_i, p_j) on patch is positioned onto (im_i, im_j)
    on image, and patch is wrapped around edges of image if necessary
    images is [N x H x W x C]
    returns [N x p_dim x p_dim x C]

    '''

    row_start = (im_i - p_i) % im_H
    col_start = (im_j - p_j) % im_W

    return np.take(
        np.take(
            images,
            range(row_start, row_start + p_dim),
            axis=1,
            mode='wrap'),
        range(col_start, col_start + p_dim),
        axis=2,
        mode='wrap')



