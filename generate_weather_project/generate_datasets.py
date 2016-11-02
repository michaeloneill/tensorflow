import numpy as np
import gzip, cPickle

from common.plotting import tile_raster_images, images_to_tuple
from common.utility_fns import sample_patches, mask_input
from PIL import Image
import h5py

import pdb



def get_train_time_patches_labels(images_train, images_val, images_test, params, channels_to_predict):
    
    # sample patches

    X_train = sample_patches(images_train, params['patch_dim'], params['n_patches_train'])
    X_val = sample_patches(images_val, params['patch_dim'], params['n_patches_val'])
    X_test = sample_patches(images_test, params['patch_dim'], params['n_patches_test'])

    # get labels before masking patches

    bins = np.linspace(0.0, 1.0, params['n_bins']+1)

    # [n_patches_train x len(channels_to_predict)] values [0,params['n_bins'])
    indices_train = np.digitize(X_train[:, params['p_i'], params['p_j'], channels_to_predict], bins, right=True)-1 
    indices_val = np.digitize(X_val[:, params['p_i'], params['p_j'], channels_to_predict], bins, right=True)-1
    indices_test = np.digitize(X_test[:, params['p_i'], params['p_j'], channels_to_predict], bins, right=True)-1

    y_train = np.zeros(shape=(params['n_patches_train'], params['n_bins'], len(channels_to_predict)))
    y_val = np.zeros(shape=(params['n_patches_val'], params['n_bins'], len(channels_to_predict)))
    y_test = np.zeros(shape=(params['n_patches_test'], params['n_bins'], len(channels_to_predict)))

    for i in xrange(params['n_patches_train']):
        for j in xrange(len(channels_to_predict)):
            y_train[i, indices_train[i, j], j] = 1

    for i in xrange(params['n_patches_val']):
        for j in xrange(len(channels_to_predict)):
            y_val[i, indices_val[i, j], j] = 1

    for i in xrange(params['n_patches_test']):
        for j in xrange(len(channels_to_predict)):
            y_test[i, indices_test[i, j], j] = 1


    # Each row is arranged [bin_0_R, bin_0_G, bin_0_B, bin_1_R, bin_1_G, bin_1_B,.....bin_256_R, bin_256_G, bin_256_B]
    y_train = y_train.reshape(-1, params['n_bins']*len(channels_to_predict))
    y_val = y_val.reshape(-1, params['n_bins']*len(channels_to_predict))
    y_test = y_test.reshape(-1, params['n_bins']*len(channels_to_predict))

    # now mask patches
    
    X_train = mask_input(X_train, params['p_i'], params['p_j'], channels_to_predict)
    X_val = mask_input(X_val, params['p_i'], params['p_j'], channels_to_predict)
    X_test = mask_input(X_test, params['p_i'], params['p_j'], channels_to_predict)

    return X_train, y_train, X_val, y_val, X_test, y_test



def generate_wind_datasets(i_filenames, o_filename_prefix, channels_to_predict, historical=False, unroll=True):

    ''' generates dataset for use at training time and one for use at test time '''
    
    data = []
    for f in i_filenames:
        d = h5py.File(f, 'r')['Dataset1']
        d = np.transpose(d, [0, 2, 3, 1]) 
        n, H, W, n_components = d.shape
        if historical:
            assert n % 4 == 0 # data should be recorded at 6 hourly intervals
            d = np.array([np.concatenate(x, 2) for x in np.split(d, n/4)]) # [n/4, H, W, 4*n_components]
        data.append(d)

    data = np.concatenate(data)
    N, H, W, n_channels = d.shape
    
    ################# IMAGE OVERFIT ##################
    data = np.array([data[0] for i in range(N)])
    ############################################

    if historical:  

        # plot first data example as 4*2 grid with rows snapshots and cols x/y components
        wind_images = Image.fromarray(tile_raster_images(
            X=data[0].transpose(2, 0, 1).reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=(4,2),
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        wind_images.save('sample_0_historical.png')

    else:

        for i in range(n_channels):
            wind_images = Image.fromarray(tile_raster_images(
                X=data[:, :, :, i].reshape(N, -1),
                img_shape=(H, W),
                tile_shape=(10,10),
                tile_spacing=(1,1),
                scale_to_unit_interval=False))
            wind_images.save('wind_channel_{}.png'.format(i))
        

    params = {
        'n_patches_train': 40000,
        'n_patches_val': 5000,
        'n_patches_test': 5000,
        'patch_dim': 10,
        'n_bins': 256,
        'p_i': 9,
        'p_j': 5
    }

    # split into train, test val

    indices = np.random.permutation(N)
    X_train = data[indices[:0.8*N]]
    X_val = data[indices[0.8*N:0.9*N]]
    X_test = data[indices[0.9*N:]]

    # save X_test for use at test time

    print 'X for use at test time has shape is {}'.format(X_test.shape)
    with open(o_filename_prefix+'_test_time.npz', 'wb') as f:
        np.savez(f, X_test_time=X_test)

    if historical:

        # plot first data example as 4*2 grid with rows snapshots and cols x/y components
        wind_images = Image.fromarray(tile_raster_images(
            X=X_test[0].transpose(2, 0, 1).reshape(-1, H*W),
            img_shape=(H, W),
            tile_shape=(4,2),
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        wind_images.save('sample_0_historical_test_time.png')

    else:
        
        for i in range(n_channels):
            wind_images = Image.fromarray(tile_raster_images(
                X=X_test[:, :, :, i].reshape(-1, H*W),
                img_shape=(H, W),
                tile_shape=(10,10),
                tile_spacing=(1,1),
                scale_to_unit_interval=False))
            wind_images.save('X_test_time_channel_{}.png'.format(i))


    # get train time patches and labels
    
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_time_patches_labels(X_train, X_val, X_test, params, channels_to_predict)

    # ############ PATCH OVERFIT ############

    # X_train = np.array([X_train[0] for i in range(X_train.shape[0])])
    # y_train = np.array([y_train[0] for i in range(y_train.shape[0])])

    # X_val = np.array([X_val[0] for i in range(X_val.shape[0])])
    # y_val = np.array([y_val[0] for i in range(y_val.shape[0])])

    # X_test = np.array([X_test[0] for i in range(X_test.shape[0])])
    # y_test = np.array([y_test[0] for i in range(y_test.shape[0])])

    # #################################################

    
    # plot masked training patches

    if historical:
        # plot first data example as 4*2 grid with rows snapshots and cols x/y components
        wind_images = Image.fromarray(tile_raster_images(
            X=X_train[0].transpose(2, 0, 1).reshape(-1, params['patch_dim']*params['patch_dim']),
            img_shape=(params['patch_dim'], params['patch_dim']),
            tile_shape=(4,2),
            tile_spacing=(1,1),
            scale_to_unit_interval=False))
        wind_images.save('sample_0_historical_masked.png')

    else:
        
        for i, j in enumerate(['x', 'y']):
            wind_images = Image.fromarray(tile_raster_images(
                X=X_train[:, :, :, i].reshape(-1, params['patch_dim']*params['patch_dim']),
                img_shape=(params['patch_dim'], params['patch_dim']),
                tile_shape=(10,10),
                tile_spacing=(1,1),
                scale_to_unit_interval=False))
            wind_images.save('wind_{}_masked_training_patches.png'.format(j))
            
    if unroll: # for use in mlp

        X_train = X_train.reshape(params['n_patches_train'], -1)
        X_val = X_val.reshape(params['n_patches_val'], -1)
        X_test = X_test.reshape(params['n_patches_test'], -1)

         
    print 'X_train and y_train shapes: {}, {}'.format(X_train.shape, y_train.shape)
    print 'X_val and y_val shapes: {}, {}'.format(X_val.shape, y_val.shape)
    print 'X_test and y_test shapes: {}, {}'.format(X_test.shape, y_test.shape)

    
    with open(o_filename_prefix+'_train_time.npz', 'wb') as f:
        np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val,
                 y_val=y_val, X_test=X_test, y_test=y_test)



if __name__=='__main__':

    ############################ wind one month #################################

    channels_to_predict = [0,1]
    
    i_filenames = ['../../data/generate_weather_project/wind/raw/wind_201401.h5']

    # mlp
    o_filename_prefix = '../../data/generate_weather_project/wind/snapshot/256_bin/wind_201401_dataset_pixel_mlp_overfit'
    generate_wind_datasets(i_filenames, o_filename_prefix, channels_to_predict, historical=False, unroll=True)

    # cnn
    o_filename_prefix = '../../data/generate_weather_project/wind/snapshot/256_bin/wind_201401_dataset_pixel_cnn_overfit.npz' 
    generate_wind_datasets(i_filenames, o_filename_prefix, channels_to_predict, historical=False, unroll=False)

    ########################### wind historical one month ######################################

    # channels_to_predict = [6,7]

    # i_filenames = ['../../data/generate_weather_project/wind/raw/wind_201401.h5']

    # # mlp
    # o_filename_prefix = '../../data/generate_weather_project/wind/historical/256_bin/wind_201401_dataset_pixel_mlp_overfit_historical'
    # generate_wind_datasets_historical(i_filenames, o_filename_prefix, channels_to_predict, historical=True, unroll=True) 

    # #cnn
    # o_filename_prefix = '../../data/generate_weather_project/wind/historical/256_bin/wind_201401_dataset_pixel_cnn_overfit_historical'
    # generate_wind_datasets_historical(i_filenames, o_filename_prefix, channels_to_predict, historical=True, unroll=False) 

    


    





    ######################## mnist #########################
    
    # generate_mnist_test_time_dataset()
    
    # o_filename = '../../data/generate_weather_project/mnist_train_time_dataset_pixel_mlp.npz'
    # generate_mnist_train_time_dataset(unroll=True, o_filename=o_filename) # mlp


    # o_filename = '../../data/generate_weather_project/mnist_train_time_dataset_pixel_cnn.npz'
    # generate_mnist_train_time_dataset(unroll=False, o_filename=o_filename) # cnn

    # o_filename = '../../data/generate_weather_project/mnist_train_time_dataset_pixel_mlp_9_5.npz'
    # generate_mnist_train_time_dataset(unroll=True, o_filename=o_filename) # mlp

    # o_filename = '../../data/generate_weather_project/mnist_train_time_dataset_pixel_cnn_9_5.npz'
    # generate_mnist_train_time_dataset(unroll=False, o_filename=o_filename) # cnn





# def generate_mnist_train_time_dataset(unroll, o_filename):

#     with gzip.open('../../data/generate_weather_project/mnist.pkl.gz', 'rb') as f:
#         train_set, val_set, test_set = cPickle.load(f)

#     # plot raw training images

#     tiled_images = Image.fromarray(tile_raster_images(
#         X=train_set[0],
#         img_shape=(28,28),
#         tile_shape=(10,10),
#         tile_spacing=(1,1),
#         scale_to_unit_interval=False))
#     tiled_images.save('mnist_images.png')

#     params = {
#         'n_patches_train': 40000,
#         'n_patches_val': 5000,
#         'n_patches_test': 5000,
#         'patch_dim': 10,
#         'n_bins': 256,
#         'n_channels': 1,
#         'p_i': 9,
#         'p_j': 5
#     }


#     X_train, X_val, X_test = get_train_time_patches(
#         train_set[0].reshape(-1, 28, 28, params['n_channels']),
#         val_set[0].reshape(-1, 28, 28, params['n_channels']),
#         test_set[0].reshape(-1, 28, 28, params['n_channels']),
#         params
#     )
    
#     # plot masked training patches

#     tiled_patches = Image.fromarray(tile_raster_images(
#         X=images_to_tuple(X_train),
#         img_shape=(patch_dim, patch_dim),
#         tile_shape=(10,10),
#         tile_spacing=(1,1),
#         scale_to_unit_interval=False))
#     tiled_patches.save('mnist_masked_training_patches.png')

#     y_train, y_val, y_test = get_train_time_labels(X_train, X_val, X_test, params)
    

#     if unroll: # for use in mlp

#         X_train = X_train.reshape(params['n_patches_train'], -1)
#         X_val = X_val.reshape(params['n_patches_val'], -1)
#         X_test = X_test.reshape(params['n_patches_test'], -1)

         
#     print 'X_train and y_train shapes: {}, {}'.format(X_train.shape, y_train.shape)
#     print 'X_val and y_val shapes: {}, {}'.format(X_val.shape, y_val.shape)
#     print 'X_test and y_test shapes: {}, {}'.format(X_test.shape, y_test.shape)

    
#     with open(o_filename, 'wb') as f:
#         np.savez(f, X_train=X_train, y_train=y_train, X_val=X_val,
#                  y_val=y_val, X_test=X_test, y_test=y_test)

        
# def generate_mnist_test_time_dataset():

#     with gzip.open('../../data/generate_weather_project/mnist.pkl.gz', 'rb') as f:
#         _, _, test_set = cPickle.load(f)

#     X_test_time = test_set[0]

#     im_dim = 28
#     n_channels = 1

#     X_test_time = X_test_time.reshape(-1, im_dim, im_dim, n_channels)

#     # zero out bottom half

#     X_test_time[:, im_dim/2:, :, :] = 0

#     print 'X_test_time shape is {}'.format(X_test_time.shape)

#     tiled_images = Image.fromarray(tile_raster_images(
#         X=images_to_tuple(X_test_time),
#         img_shape=(28,28),
#         tile_shape=(10,10),
#         tile_spacing=(1,1),
#         scale_to_unit_interval=False))
#     tiled_images.save('mnist_images_test_time.png')


#     with open('../../data/generate_weather_project/mnist_test_time_dataset.npz', 'wb') as f:
#         np.savez(f, X_test_time=X_test_time)
