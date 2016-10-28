import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import sqlite3
import pdb
import os


def get_long_lat_from_weather_db(forecast_site_codes):

    ''' returns locations as list n_loc *[long,lat] '''
    
    conn = sqlite3.connect('../../data/met_office_observations/weather.db')
    c = conn.cursor()

    sql = '''SELECT
    FORECASTSITECODE,
    LONGITUDE, 
    LATITUDE 
    FROM WEATHER WHERE
    FORECASTSITECODE IN ({seq})
    GROUP BY FORECASTSITECODE
    '''.format(seq=','.join(['?']*len(forecast_site_codes)))

    c.execute(sql, forecast_site_codes)

    all_rows = c.fetchall()

    locations = np.array([(float(row[0]), float(row[1]), float(row[2])) for row in all_rows])

    # check order correct

    assert np.all(locations[:, 0] == forecast_site_codes)
    
    conn.commit()
    conn.close()

    # do not return site code
    return locations[:, 1:]


def plot_wind_map_ground_vs_preds(X_ground, X_preds, locations, bbox, n_plots, results_dir, scale=None, scale_units=None):

    ''' X_ground and X_preds are [nExamples, n_loc, 2]
    locations is list of  n_loc * [long, lat] 
    bbox is bounding box for plot 
    '''
    
    fig, axes = plt.subplots(n_plots, 2)
    
    for i, (x_ground, x_preds, ax_row) in enumerate(zip(X_ground[:n_plots], X_preds[:n_plots], axes)):
        if i==0:
            ax_row[0].set_title('Ground Truth')
            ax_row[1].set_title('Predictions')
        plot_wind_map(x_ground, locations, bbox, ax_row[0], scale=scale, scale_units=scale_units)
        plot_wind_map(x_preds, locations, bbox, ax_row[1], scale=scale, scale_units=scale_units)

    fig.savefig(results_dir + 'wind_map_ground_vs_preds.png')


def plot_wind_map(obs, locations, bbox, ax, scale=None, scale_units=None):
    
    ''' obs is [nLocations, 2] 
    locations is list of n_loc * [long, lat] 
    bbox is bounding box for basemap plot
    '''
    fig, axes = plt.subplots(4, 4)

    axes = axes.flatten()

    m = Basemap(
        projection='mill', lon_0=bbox['lon'], lat_0=bbox['lat'],
        llcrnrlon=bbox['ll_lon'], llcrnrlat=bbox['ll_lat'],
        urcrnrlon=bbox['ur_lon'], urcrnrlat=bbox['ur_lat'], ax=ax)

    # Convert locations to x/y coordinates.
    lons, lats = zip(*locations)
    x, y = m(lons, lats)

    # Avoid border around map.
    m.drawmapboundary(fill_color='#ffffff', linewidth=.0)
    m.drawcoastlines()

    # plot winds
    m.quiver(x, y, obs[:, 0], obs[:, 1], scale=scale, scale_units=scale_units)
    

def plot_pressure_map_ground_vs_preds(X_ground, X_preds, locations, bbox, n_plots, results_dir):

    ''' X_ground and X_preds are [nExamples, n_loc]
    locations is list of  n_loc * [long, lat] 
    bbox is bounding box for plot 
    '''

    fig, axes = plt.subplots(n_plots, 2)
    
    for i, (x_ground, x_preds, ax_row) in enumerate(zip(X_ground[:n_plots], X_preds[:n_plots], axes)):
        if i==0:
            ax_row[0].set_title('Ground Truth')
            ax_row[1].set_title('Predictions')
        plot_pressure_map(x_ground, locations, bbox, ax_row[0])
        plot_pressure_map(x_preds, locations, bbox, ax_row[1])

    fig.savefig(results_dir + 'pressure_map_ground_vs_preds.png')



def plot_pressure_map(obs, locations, bbox, ax):

    ''' obs is 1d array of length n_loc 
    locations is list of  n_loc * [long, lat] 
    bbox is bounding box for plot 
    ax is the axes to plot onto '''

    m = Basemap(
        projection='mill', lon_0=bbox['lon'], lat_0=bbox['lat'],
        llcrnrlon=bbox['ll_lon'], llcrnrlat=bbox['ll_lat'],
        urcrnrlon=bbox['ur_lon'], urcrnrlat=bbox['ur_lat'], ax=ax)

    # Convert locations to x/y coordinates for plotting
    lons, lats = zip(*locations)
    x, y = m(lons, lats)

    # create gridpoints to interpolate over
    xi = np.linspace(np.floor(min(x)), np.ceil(max(x)), 100)
    yi = np.linspace(np.floor(min(y)), np.ceil(max(y)), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Avoid border around map.
    m.drawmapboundary(fill_color='#ffffff', linewidth=.0)
    m.drawcoastlines()

    zi = griddata((x,y), obs, (grid_x, grid_y), method='cubic')

    m.contourf(grid_x, grid_y, zi, cmap=plt.cm.jet)
    m.scatter(x, y, marker='x', color='r', alpha=0.8)


    
def plot_locations_map(locations_from, locations_to, bbox, results_dir):
    '''
    locations_from is n_loc_from * [long, lat] 
    locations_to is n_loc_to * [long, lat] 
    bbox is bounding box for plot 
    '''

    fig = plt.figure()
    
    m = Basemap(
        projection='mill', lon_0=bbox['lon'], lat_0=bbox['lat'],
        llcrnrlon=bbox['ll_lon'], llcrnrlat=bbox['ll_lat'],
        urcrnrlon=bbox['ur_lon'], urcrnrlat=bbox['ur_lat'])

    # Convert locations to x/y coordinates for plotting
    lons_from, lats_from = zip(*locations_from)
    x_from, y_from = m(lons_from, lats_from)

    lons_to, lats_to = zip(*locations_to)
    x_to, y_to = m(lons_to, lats_to)

    # Avoid border around map.
    m.drawmapboundary(fill_color='#ffffff', linewidth=.0)
    m.drawcoastlines()

    m.scatter(x_from, y_from, marker='x', color='r', alpha=0.8, label='interpolate_from')
    m.scatter(x_to, y_to, marker='o', color= 'b', alpha=0.8, label='interpolate_to')
    plt.legend()
    
    plt.savefig(results_dir + 'locations_map.png')

    
