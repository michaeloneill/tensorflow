import tensorflow as tf
import numpy as np
import pdb
from PIL import Image

from plotting import tile_raster_images, images_to_tuple
from utility_fns import build_train_graph, get_loss 


from model_components import build_mlp


def build_skynet_mlp(params):

    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape'],
                           name='x')

        y_ = tf.placeholder(tf.float32,
                            shape=params['targets_shape'],
                            name='y_')

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                
    with tf.variable_scope('mlp'):
        mlp_output = build_mlp(x, dropout_keep_prob, params['mlp'])
            

    with tf.name_scope('loss'):
        loss = get_loss(mlp_output, y_) 
    tf.scalar_summary('loss', loss)

    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'output': mlp_output
    }
    
    return model



def generate_predictions(observations, img_shape, params, model, sess, results_dir):

    predictions = sess.run(model['output'],
        feed_dict = {
            model['x']: observations,
            model['dropout_keep_prob']:params['dropout_keep_prob']
        }
    )

    print 'max value in predictions is {}'.format(predictions.max())
    print 'min value in predictions is {}'.format(predictions.min())

    assert predictions.shape[0]==observations.shape[0]

    # plot predictions as images

    predictions_images = Image.fromarray(tile_raster_images(
        X=predictions[:100],
        img_shape=img_shape,
        tile_shape=(10,10),
        tile_spacing=(1,1),
        scale_to_unit_interval=False))
    predictions_images.save(results_dir+'predictions.png')
