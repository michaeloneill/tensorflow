import tensorflow as tf
import numpy as np
import pdb
import sys

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



# def run_val(val_set, params, model, sess, miniBatchIndex):

#     return sess.run([model['loss']], feed_dict = {
#         model['x']: val_set[0][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#         model['y_']: val_set[1][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#         model['dropout_keep_prob']:0.0
#     }
#     )

# loss_val = np.mean(
#     [run_val(val_set, params, model, sess, i) for i in xrange(nBatchVal)]
# )



# def run_test(test_set, params, model, sess, miniBatchIndex):

#     return sess.run([model['loss']], feed_dict = {
#         model['x']: test_set[0][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#         model['y_']: test_set[1][miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#         model['dropout_keep_prob']:0.0
#     }
#     )

# loss_test = np.mean(
#         [run_test(val_set, params, model, sess, i) for i in xrange(nBatchTest)]
# )


def train(train_set, val_set, test_set, params, model, sess, results_dir):

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(results_dir+'logs/train', sess.graph)
    val_writer = tf.train.SummaryWriter(results_dir+'logs/val', sess.graph)
    
    nBatchTrain = train_set[0].shape[0]/params['miniBatchSize']
    nBatchVal = val_set[0].shape[0]/params['miniBatchSize']
    nBatchTest = val_set[0].shape[0]/params['miniBatchSize']

    agg_loss_train = 0.0
    print 'starting training...'
    for epoch in range(params['epochs']):
        for miniBatchIndex in range(nBatchTrain):
            iteration = epoch*nBatchTrain + miniBatchIndex

            _, loss_train, summary = run_train(train_set, params, model, sess, miniBatchIndex, merged)
            train_writer.add_summary(summary, iteration) 
            
            agg_loss_train += loss_train
            
            if (iteration+1)%params['monitor_frequency'] == 0:
                print 'train loss per minibatch for epoch {0}, minibatch {1}/{2} is: {3:.2f}'.format(
                    epoch+1, miniBatchIndex+1, nBatchTrain, agg_loss_train/(iteration+1))
                loss_val, summary = run_val(val_set, params, model, sess, merged) 
                val_writer.add_summary(summary, iteration) 
                print 'correpsonding val loss is: {:.2f}'.format(loss_val)

    loss_test = run_test(val_set, params, model, sess) 
    print 'Training complete. Final loss on test set is: {:.2f}'.format(loss_test)



def unit_scale(X):
        """ Scales all values in X to be between 0 and 1 """

        X = X.astype(np.float32) # copies by default
        X -= X.min()
        X *= 1.0 / X.max()
        return X
    


def sample_patches(images, patch_dim, num_patches):

    ''' 
    images is [N x H x W x C] 
    returns [num_patches x patch_dim x patch_dim x C] '''

    N = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3] # channels

    patches = np.zeros((num_patches, patch_dim, patch_dim, C))

    # coordinates limit for top left of patch 
    max_row_start = H - patch_dim
    max_col_start = W - patch_dim

    for i in xrange(num_patches):
        row_start = np.random.randint(max_row_start + 1)
        col_start = np.random.randint(max_col_start + 1)
        im_idx = np.random.randint(N)
        patches[i, :, :, :] = images[im_idx, row_start : row_start + patch_dim, col_start : col_start + patch_dim, :]

    return patches


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



################################ SKYNET w/ base images ###############################################################
# def train(inpts, targets, base_images, params, model, sess, results_dir):

#     merged = tf.merge_all_summaries()
#     train_writer = tf.train.SummaryWriter(results_dir+'logs/train', sess.graph)
    
#     nBatchTrain = inpts.shape[0]/params['miniBatchSize']

#     monitorCost = 0.0
#     print 'starting training...'
#     for epoch in range(params['epochs']):
#         for miniBatchIndex in range(nBatchTrain):
#             iteration = epoch*nBatchTrain + miniBatchIndex
#             _, loss, summary = sess.run([model['train'], model['loss'], merged],
#                     feed_dict = {
#                         model['x']: inpts[miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#                         model['y_']: targets[miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#                         model['dropout_keep_prob']:params['dropout_keep_prob'],
#                         model['base_images']: base_images[miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#                     }
#             )
            
#             train_writer.add_summary(summary, iteration)
            
#             monitorCost += loss
            
#             if (iteration+1)%params['monitor_frequency'] == 0:
#                 print 'loss per minibatch for epoch {0}, minibatch {1}/{2} is: {3:.2f}'.format(
#                     epoch+1, miniBatchIndex+1, nBatchTrain, monitorCost/(iteration+1))
#########################################################################################################################




################################################# SKYNET w/o base images ###########################################################
# def train(inpts, targets, params, model, sess, results_dir):

#     merged = tf.merge_all_summaries()
#     train_writer = tf.train.SummaryWriter(results_dir+'logs/train', sess.graph)
    
#     nBatchTrain = inpts.shape[0]/params['miniBatchSize']

#     monitorCost = 0.0
#     print 'starting training...'
#     for epoch in range(params['epochs']):
#         for miniBatchIndex in range(nBatchTrain):
#             iteration = epoch*nBatchTrain + miniBatchIndex
#             _, loss, summary = sess.run([model['train'], model['loss'], merged],
#                     feed_dict = {
#                         model['x']: inpts[miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#                         model['y_']: targets[miniBatchIndex*params['miniBatchSize']:(miniBatchIndex+1)*params['miniBatchSize']],
#                         model['dropout_keep_prob']:params['dropout_keep_prob']
#                     }
#             )
            
#             train_writer.add_summary(summary, iteration)
            
#             monitorCost += loss
            
#             if (iteration+1)%params['monitor_frequency'] == 0:
#                 print 'loss per minibatch for epoch {0}, minibatch {1}/{2} is: {3:.2f}'.format(
#                     epoch+1, miniBatchIndex+1, nBatchTrain, monitorCost/(iteration+1))
#####################################################################################################################################





# def build_train_graph(loss, params):
#     with tf.name_scope('train'):
#         train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
#         return train_step
