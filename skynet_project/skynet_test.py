import tensorflow as tf
import numpy as np
import pdb
from PIL import Image

from plotting import tile_raster_images, images_to_tuple
from utility_fns import train, build_train_graph, get_loss 

from model_components import build_cnn, build_mlp, build_dcnn

def build_ae(params):

        
    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape'],
                           name='x')

        # image outputs
        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape'],
                            name='y_')

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        

    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, params['mlp'])

            
    with tf.variable_scope('dmlp'):
        dmlp_output = build_mlp(mlp_output, dropout_keep_prob, params['dmlp'])

    with tf.name_scope('loss'):
        loss = get_loss(dmlp_output, y_) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': dmlp_output
    }
    
    return model



def build_cae(params):

        
    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape'],
                           name='x')

        # image outputs
        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape'],
                            name='y_')

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')


    with tf.variable_scope('cnn'):
        cnn_output = build_cnn(x, dropout_keep_prob, params['cnn'])

            
    with tf.variable_scope('dcnn'):
        dcnn_output = build_dcnn(cnn_output, dropout_keep_prob, params['dcnn'])

    with tf.name_scope('loss'):
        flattened_dim = [-1,
                         params['inpt_shape'][1]*params['inpt_shape'][2]*params['inpt_shape'][3]]
        dcnn_output_flattened = tf.reshape(dcnn_output, flattened_dim)
        targets_flattened = tf.reshape(y_, flattened_dim)
        loss = get_loss(dcnn_output_flattened, targets_flattened) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': dcnn_output
    }
    
    return model



def generate_images_from_model(input_images, img_shape, params, model, sess, results_dir):

    generated_images = sess.run(model['output'],
        feed_dict = {
            model['x']: input_images,
            model['dropout_keep_prob']: params['dropout_keep_prob']
        }
    )


    # manually calculate loss

    loss = np.mean(np.sum(np.square(generated_images-input_images), axis=1))
    print 'average loss per minibatch is: {:.2f}'.format(loss*params['miniBatchSize'])


    # plot input and generated images
    
    input_images = input_images.reshape((-1,) + img_shape[1:])
    generated_images = generated_images.reshape((-1,)+img_shape[1:])

    tiled_input_images = Image.fromarray(tile_raster_images(
        X=images_to_tuple(input_images[:100]),
        img_shape=img_shape[1:3],
        tile_shape=(10,10),
        tile_spacing=(1,1)))
    tiled_input_images.save(results_dir+'input_images.png')

    
    tiled_generated_images = Image.fromarray(tile_raster_images(
        X=images_to_tuple(generated_images[:100]),
        img_shape=img_shape[1:3],
        tile_shape=(10,10),
        tile_spacing=(1,1)))
    tiled_generated_images.save(results_dir+'generated_images.png')


def main():


    data_filename='../data/skynet_trial_dataset_30x30x3.npz'

    results_dir = input('Enter results directory: ')

    data = np.load(data_filename)
    images = data['images'].astype(np.float32)
    

    ################### TEST 1 ############################

    if 0:

        # simple autoencoder 1 fc layer

        params_mlp = {
            'inpt_dim': 30*30*3,
            'num_outputs': [100],
            'activations':['sigmoid'],
            'dropout': [False]
        }

        params_dmlp = {
            'inpt_dim': 100,
            'num_outputs': [30*30*3],
            'activations': ['sigmoid'],
            'dropout': [False]
        }


        params_train = {
            'miniBatchSize': 20,
            'epochs': 100,
            'learning_rate':0.01,
            'dropout_keep_prob': 0.5,
            'monitor_frequency': 10
        }

        params = {
            'mlp': params_mlp,
            'dmlp': params_dmlp,
            'train': params_train,
            'inpt_shape': [None, 30*30*3],
            'device':'/gpu:1',
            'results_dir': results_dir
        }



        # # use first two channels of first image repeated
        # r = np.copy(images[0])
        # r[:, :, [1,2]]=0
        # g = np.copy(images[0])
        # g[:, :, [0,2]]=0
        # rg = np.array([r,g])
        # images = np.concatenate(tuple(rg for i in range(images.shape[0]/2)))
        # print images.shape

        # # use first image repeated
        # images = np.array([images[0] for i in range(images.shape[0])])

        # # use first 2 images repeated
        # two = images[[17, 27]]
        # images = np.concatenate(tuple(two for i in range(images.shape[0]/2)))

        images = images.reshape((-1, params['inpt_shape'][1]))
        print images.shape
    
        model = build_ae(params)
    
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        train(images, images, params['train'], model, sess, params['results_dir'])

        generate_images_from_model(images, (None, 30, 30, 3), params['train'], model,
                                   sess, params['results_dir'])
    

    ################### TEST 2 ##############################

    if 1:
        
        # convolutional autoencoder

        params_cnn = {
            'kernel_size': [[5,5], [5,5]],
            'num_outputs': [6, 12],
            'activations': ['sigmoid', 'sigmoid'],
            'fc_params':{'inpt_dim': 30*30*12, 'num_outputs': [100], 'activations': ['sigmoid'], 'dropout': [False]}
        }

        params_dcnn = {
            'fc_params':{'inpt_dim': 100, 'num_outputs': [30*30*12], 'activations':['sigmoid'], 'dropout':[False]},
            'inpt_reshape': [30, 30, 12],
            'kernel_size': [[5,5], [5,5]],
            'num_outputs': [6, 3],
            'activations': ['sigmoid', 'sigmoid']
        }

        params_train = {
            'miniBatchSize': 20,
            'epochs': 100,
            'learning_rate':0.01,
            'dropout_keep_prob': 0.5,
            'monitor_frequency': 10
        }

        params = {
            'cnn': params_cnn,
            'dcnn': params_dcnn,
            'train': params_train,
            'inpt_shape': [None, 30, 30, 3],
            'device':'/gpu:1',
            'results_dir': results_dir
        }

        # # use first image repeated
        # images = np.array([images[0] for i in range(images.shape[0])])

        # # use first two images repeated
        # two = images[[17, 27]]
        # images = np.concatenate(tuple(two for i in range(images.shape[0]/2)))

        # # use first two channels of first image repeated
        # r = np.copy(images[0])
        # r[:, :, [1,2]]=0
        # g = np.copy(images[0])
        # g[:, :, [0,2]]=0
        # rg = np.array([r,g])
        # images = np.concatenate(tuple(rg for i in range(images.shape[0]/2)))
        # print images.shape

        model = build_cae(params)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        train(images, images, params['train'], model, sess, params['results_dir'])

        generate_images_from_model(images[:100], images.shape, params['train'], model,
                                   sess, params['results_dir'])



    ########################################################



if __name__=='__main__':

    main()    
