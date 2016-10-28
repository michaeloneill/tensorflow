import tensorflow as tf
import numpy as np
import pdb
from PIL import Image

from plotting import tile_raster_images, images_to_tuple
from utility_fns import train, build_train_graph, get_loss 
import os


from model_components import build_cnn, build_mlp, build_dcnn


def build_skynet(params):

        
    with tf.name_scope('input'):

        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['observations'],
                           name='x')

        base_images = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['images'],
                           name='base_images')

        # image outputs
        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['images'],
                            name='y_')


    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    with tf.variable_scope('cnn'):
        cnn_output = build_cnn(base_images, dropout_keep_prob, params['cnn'])
                
    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, params['mlp'])
            
    with tf.variable_scope('combine'):
        #images_observations = tf.concat(1, [cnn_output, mlp_output])
        images_observations = cnn_output + mlp_output

    with tf.variable_scope('dcnn'):
        dcnn_output = build_dcnn(images_observations, dropout_keep_prob, params['dcnn'])

    with tf.name_scope('loss'):
        flattened_dim = [-1,
                         params['inpt_shape']['images'][1]*params['inpt_shape']['images'][2]*params['inpt_shape']['images'][3]]
        dcnn_output_flattened = tf.reshape(dcnn_output, flattened_dim)
        targets_flattened = tf.reshape(y_, flattened_dim)
        loss = get_loss(dcnn_output_flattened, targets_flattened) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'base_images': base_images,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': dcnn_output
    }
    
    return model


def generate_images_from_model(observations, base_images, img_shape, params, model, sess, results_dir):

    generated_images = sess.run(model['output'],
        feed_dict = {
            model['x']: observations,
            model['base_images']: base_images,
            model['dropout_keep_prob']:params['dropout_keep_prob']
        }
    )

    print generated_images.max()
    print generated_images.min()

    assert generated_images.shape[0]==observations.shape[0]==base_images.shape[0]

    # plot images
    
    tiled_images = Image.fromarray(tile_raster_images(
        X=images_to_tuple(generated_images[:100]),
        img_shape=img_shape[1:3],
        tile_shape=(10,10),
        tile_spacing=(1,1),
        scale_to_unit_interval=False))
    tiled_images.save(results_dir+'generated_images.png')


    

def main():

    results_dir = input('Enter results directory: ')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    params_cnn = {
        'kernel_size': [[5, 5], [5, 5]],
        'num_outputs': [6, 12],
        'activations': ['sigmoid', 'sigmoid'],
        'fc_params':{'num_outputs': [100], 'activations': ['sigmoid'], 'dropout':[False]}
    }

    params_mlp = {
        'num_outputs': [100],
        'activations': ['sigmoid'],
        'dropout': [False]
    }

    
    params_dcnn = {
        'fc_params':{'num_outputs': [30*30*12], 'activations': ['sigmoid'], 'dropout':[False]},
        'raster_shape': [30, 30, 12],
        'kernel_size': [[5,5], [5,5]],
        'num_outputs': [6, 3],
        'activations': ['sigmoid', 'sigmoid']
    }

    params_train = {
        'miniBatchSize': 20,
        'epochs':10000,
        'learning_rate':0.001,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'cnn': params_cnn,
        'mlp': params_mlp,
        'dcnn': params_dcnn,
        'train': params_train,
        'inpt_shape': {'images': [None, 30, 30, 3], 'observations': [None, 11]},
        'device':'/gpu:1',
        'results_dir': results_dir
    }


    data_filename='../data/hampstead_dataset.npz'

    data = np.load(data_filename)
    
    images = data['images'].astype(np.float32)
    observations = data['observations'].astype(np.float32)
    base_images = np.array([images[0] for i in range(images.shape[0])])


    
    # just use first image repeated and no observations
    # images = np.array([images[0] for i in range(images.shape[0])])
    # observations = np.zeros((observations.shape))

    # use first image at the start of each day as the base image
    base_images = np.concatenate(
        tuple(
            np.array([images[9*i] for j in range(9)])
            for i in range(images.shape[0]/9)
        )
    )


    # plot base images
    
    image = Image.fromarray(tile_raster_images(
        X=images_to_tuple(base_images[:100]),
        img_shape=params['inpt_shape']['images'][1:3],
        tile_shape=(10,10)))
    image.save(params['results_dir']+'base_imgs.png')

        
    model = build_skynet(params)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    print images.shape
    print observations.shape
    
    train(observations, images,
          base_images, params['train'], model,
          sess, params['results_dir'])

    generate_images_from_model(observations, base_images,
                               images.shape, params['train'], model,
                               sess, params['results_dir'])


    target_images = Image.fromarray(tile_raster_images(
        X=images_to_tuple(images[:100]),
        img_shape=images.shape[1:3],
        tile_shape=(10,10),
        tile_spacing=(1,1),
        scale_to_unit_interval=False))
    target_images.save(params['results_dir']+'target_images.png')


    

if __name__=='__main__':

    main()    

    


        

