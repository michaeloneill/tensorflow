import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from common.model_components import build_mlp_model
from common.utility_fns import train
from plotting import get_long_lat_from_weather_db, plot_pressure_map_ground_vs_preds, plot_locations_map

import os
import pdb


def test_model(X_test, y_test, locations_test, bbox, params, model, sess, n_plots, results_dir):

    predictions = sess.run(model['output'],
        feed_dict = {
            model['x']: X_test,
            model['dropout_keep_prob']: params['dropout_keep_prob']
        }
    )

    # use when interolating at single test site (e.g. Heathrow)
    fig = plt.figure()
    n = X_test.shape[0]
    plt.title('Interpolating pressure at Heathrow from 11 nearby sites')
    plt.plot([i for i in range(n)], y_test, c='r', label='ground truth')
    plt.xlabel('observation number')
    plt.ylabel('normalised pressure')
    plt.plot([i for i in range(n)], predictions, c='b', label='predictions')
    plt.legend()
        
    plt.savefig(results_dir + 'pressure_ground_vs_preds.png')

    # use when interpolating at at least 4 test sites
    # plot_pressure_map_ground_vs_preds(y_test, predictions,
    #                                   locations_test, bbox,
    #                                   n_plots, results_dir)


def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    params_mlp = {
        'num_outputs': [100, 100, 1],
        'activations': ['sigmoid', 'sigmoid', 'sigmoid'],
        'dropout': [False, False, False]
    }


    params_train = {
        'miniBatchSize': 20,
        'epochs':5,
        'learning_rate':0.001,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 10,
        'momentum': 0.9,
        'grad_clip': 5,
        'loss_fn': 'mean_squared'
    }

    params = {
        'mlp': params_mlp,
        'train': params_train,
        'inpt_shape': {'x': [None, 11], 'y_': [None, 1]},
        'device':'/gpu:1',
        'results_dir': results_dir
    }

    data_filename = '../../data/interpolation_project/met_pressure_interp_dataset.npz'

    data = np.load(data_filename)

    train_set = [data['X_train'], data['Y_train']]
    val_set = [data['X_val'], data['Y_val']]
    test_set = [data['X_test'], data['Y_test']]

    model = build_mlp_model(params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    train(train_set, val_set, test_set, params['train'], model, sess, results_dir)


    # visualise predictions of trained model

    sites = sorted([3772, 3672, 3684, 3660, 3768, 3781, 3761, 3658, 3649, 3749, 3769, 3882])
    target_site_codes = [3772] # heathrow

    locations_from = get_long_lat_from_weather_db([i for i in sites if i not in target_site_codes])
    locations_to = get_long_lat_from_weather_db(target_site_codes)
    
    # adapted from GeoPlanet Explorer (UK)
    bbox = {
        'lon': -5.23636,
        'lat': 53.866772,
        'll_lon': -1.65073,
        'll_lat': 50.16209,
        'ur_lon': 1.76334,
        'ur_lat': 52.860699
    }

    plot_locations_map(locations_from, locations_to, bbox, params['results_dir'])

    n_plots = 4
    
    test_model(test_set[0][:100], test_set[1][:100],
               locations_to, bbox,
               params['train'], model,
               sess, n_plots,
               params['results_dir'])


if __name__=='__main__':
    main()
    
