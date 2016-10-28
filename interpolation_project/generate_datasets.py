import datetime
import numpy as np
import sqlite3
import os
import csv
import pdb

def daterange(start_date, end_date):
    # end_date should be one past last date to include
    for n in range(int((end_date-start_date).days)):
        yield start_date + datetime.timedelta(n)


def generate_met_winds_interp_dataset(start_date, end_date, sites, target_site, start_time=0, end_time=24):


    conn = sqlite3.connect('../../data/met_office_observations/weather.db')
    c = conn.cursor()

    X = []
    Y = []

    for date in daterange(start_date, end_date):
        for time in range(start_time, end_time):

            ignore_time = False
            
            x = []
            y = []
            
            sql = ''' SELECT FORECASTSITECODE, 
            WINDDIRECTION, 
            WINDSPEED
            FROM WEATHER 
            WHERE OBSERVATIONDATE == ?
            AND OBSERVATIONTIME ==  ?
            AND FORECASTSITECODE IN ({seq})
            ORDER BY FORECASTSITECODE
            '''.format(seq=','.join(['?']*len(sites)))
            
            bindings = [date.strftime('%Y-%m-%dT00:00:00'), time] + sites
            c.execute(sql, bindings)
            
            all_rows = c.fetchall()

            if [row[0] for row in all_rows] != sites:
                # either missing or duplicate entries
                print 'ignoring {} {}'.format(date, time)
                continue

            for row in all_rows:
                if any(row[i] for i in range(1, len(row))) < 0:
                    print 'negative in {} {}'.format(date, time)
                    ignore_time=True
                    break
                try:
                    entry = [row[2]*np.cos((row[1]-1)*2*np.pi/16), -row[2]*np.sin((row[1]-1)*2*np.pi/16)]
                except ValueError, arg:
                    print 'could not convert value in {} {} to float'.format(date, time)
                    print arg
                    ignore_time=True
                    break
                except:
                    print 'unknown error'
                    ignore_time=True
                    break
                if row[0]==target_site:
                    y = entry
                else:
                    assert row[0] in sites
                    x.append(entry)

            if not ignore_time:
                X.append(x)
                Y.append(y)

    X = np.array(X, dtype=np.float32)
    # reshape
    X = X.reshape(-1, X.shape[1]*X.shape[2]) 
    Y = np.array(Y, dtype=np.float32)

    maximum = max([np.max(X), np.max(Y)])
    minimum = min([np.min(X), np.min(Y)])

    # scale between 1 and -1
    X = (X-minimum)/(maximum-minimum)*2-1
    Y = (Y-minimum)/(maximum-minimum)*2-1


    assert X.max() <= 1 and Y.max() <=1 and X.min() >= -1.0 and Y.min() >= -1.0

    
    # split into train, test, val

    nSamples = X.shape[0]

    indices = np.random.permutation(nSamples)

    X_train = X[indices[:0.8*nSamples]]
    Y_train = Y[indices[:0.8*nSamples]] 

    X_val = X[indices[0.8*nSamples:0.9*nSamples]]
    Y_val = Y[indices[0.8*nSamples:0.9*nSamples]] 

    X_test = X[indices[0.9*nSamples:]]
    Y_test = Y[indices[0.9*nSamples:]] 

    print 'X_train shape: {}'.format(X_train.shape)
    print 'Y_train shape: {}'.format(Y_train.shape)

    print 'X_val shape: {}'.format(X_val.shape)
    print 'Y_val shape: {}'.format(Y_val.shape)

    print 'X_test shape: {}'.format(X_test.shape)
    print 'Y_test shape: {}'.format(Y_test.shape)

    
    with open('../../data/interpolation_project/met_winds_interp_dataset_reshaped_scaled.npz', 'wb') as f:
        np.savez(f, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)

    

def generate_met_pressure_interp_dataset(start_date, end_date, sites, target_site, start_time=0, end_time=24):

    conn = sqlite3.connect('../../data/met_office_observations/weather.db')
    c = conn.cursor()

    X = []
    Y = []
    
    for date in daterange(start_date, end_date):
        for time in range(start_time, end_time):

            ignore_time = False
            
            x = []
            y = None
            
            sql = ''' SELECT FORECASTSITECODE, PRESSURE FROM WEATHER 
            WHERE OBSERVATIONDATE == ?
            AND OBSERVATIONTIME ==  ?
            AND FORECASTSITECODE IN ({seq})
            ORDER BY FORECASTSITECODE
            '''.format(seq=','.join(['?']*len(sites)))
            
            bindings = [date.strftime('%Y-%m-%dT00:00:00'), time] + sites
            c.execute(sql, bindings)
            
            all_rows = c.fetchall()

            if [row[0] for row in all_rows] != sites:
                # either missing or duplicate entries
                print 'ignoring {} {}'.format(date, time)
                continue

            for row in all_rows:
                try:
                    float(row[1])
                except ValueError, arg:
                    print '{} {}: could not convert {} to float'.format(date, time, row[1])
                    ignore_time=True
                    break
                if row[1] < 0:
                    print '{} {}: {} is negative'.format(date, time, row[1])
                    ignore_time=True
                    break
                if row[0]==target_site:
                    y = row[1]
                else:
                    assert row[0] in sites
                    x.append(row[1])

            if not ignore_time:
                X.append(x)
                Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)[:, None] # make column vec

    
    maximum = max([np.max(X), np.max(Y)])
    minimum = min([np.min(X), np.min(Y)])

    X = (X-minimum)/(maximum-minimum)
    Y = (Y-minimum)/(maximum-minimum)


    assert X.max() <= 1 and Y.max() <=1 and X.min() >=0 and Y.min() >= 0

    
    # split into train, test, val

    nSamples = X.shape[0]

    indices = np.random.permutation(nSamples)

    X_train = X[indices[:0.8*nSamples]]
    Y_train = Y[indices[:0.8*nSamples]] 

    X_val = X[indices[0.8*nSamples:0.9*nSamples]]
    Y_val = Y[indices[0.8*nSamples:0.9*nSamples]] 

    X_test = X[indices[0.9*nSamples:]]
    Y_test = Y[indices[0.9*nSamples:]] 

    print 'X_train shape: {}'.format(X_train.shape)
    print 'Y_train shape: {}'.format(Y_train.shape)

    print 'X_val shape: {}'.format(X_val.shape)
    print 'Y_val shape: {}'.format(Y_val.shape)

    print 'X_test shape: {}'.format(X_test.shape)
    print 'Y_test shape: {}'.format(Y_test.shape)

    
    with open('../../data/interpolation_project/met_pressure_interp_dataset.npz', 'wb') as f:
        np.savez(f, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
        

if __name__=='__main__':

    start_date = datetime.datetime(2012, 01, 01)
    end_date = datetime.datetime(2016, 01, 01) # one past last date
    start_time = 9
    end_time = 18 # one past last time

    sites = sorted([3772, 3672, 3684, 3660, 3768, 3781, 3761, 3658, 3649, 3749, 3769, 3882])
    target_site = 3772 # heathrow

    # generate_met_pressure_interp_dataset(start_date, end_date, sites, target_site, start_time, end_time)

    generate_met_winds_interp_dataset(start_date, end_date, sites, target_site, start_time, end_time)

    
















# def generate_met_pressure_interp_dataset(start_date, end_date, start_time=0, end_time=24):

#     conn = sqlite3.connect('../../data/met_office_observations/weather.db')
#     c = conn.cursor()

#     sites_b = [3712, 3672, 3684, 3660, 3768, 3781, 3761, 3658, 3649, 3749, 3769, 3882]
#     obs_a = []
#     obs_b = []
#     remove_sites = []

#     sql = ''' SELECT DISTINCT FORECASTSITECODE 
#     FROM WEATHER 
#     ORDER BY FORECASTSITECODE '''

#     c.execute(sql)
#     all_rows = c.fetchall()
#     sites = [row[0] for row in all_rows]
#     print 'There are {} sites'.format(len(sites))
    
#     for date in daterange(start_date, end_date):
#         for time in range(start_time, end_time):
                        
#             sql = ''' SELECT FORECASTSITECODE, PRESSURE FROM WEATHER 
#             WHERE OBSERVATIONDATE == ?
#             AND OBSERVATIONTIME ==  ?
#             '''
#             bindings = [date.strftime('%Y-%m-%dT00:00:00'), time]
#             c.execute(sql, bindings)
            
#             all_rows = c.fetchall()

#             fetched = [row[0] for row in all_rows]
#             if len(fetched)==0:
#                 continue

#             remove_sites.extend([i for i in sites if i not in fetched])

#     remove_sites = sorted(list(set(remove_sites)))
#     sites_b = [i for i in sites_b if i not in remove_sites]
    
#     print 'removing sites {}'.format(remove_sites)
#     print 'sites b is {}'.format(sites_b)

#     for date in daterange(start_date, end_date):
#         for time in range(start_time, end_time):
            
#             example_a = []
#             example_b = []
            
#             sql = ''' SELECT FORECASTSITECODE, PRESSURE FROM WEATHER 
#             WHERE OBSERVATIONDATE == ?
#             AND OBSERVATIONTIME ==  ?
#             WHERE FORECASTSITECODE NOT IN ({seq})
#             ORDER BY FORECASTSITECODE ''' .format(seq=','.join(['?']*len(remove_sites)))

#             bindings = [date.strftime('%Y-%m-%dT00:00:00'), time] + remove_sites
#             c.execute(sql, bindings)
            
#             all_rows = c.fetchall()

#             fetched = [row[0] for row in all_rows]

#             if len(fetched)==0:
#                 continue
            
#             assert fetched==sites
            
#             for row in all_rows:
#                 if row[0] in sites_b:
#                     example_b.append(row[1])
#                 else:
#                     example_a.append(row[1])

#             obs_a.append(example_a)
#             obs_b.append(example_b)

#     print 'obs A shape is {}'.format(obs_a.shape)
#     print 'obs B shape is {}'.format(obs_b.shape)


#     with open('../../data/met_pressure_interp_dataset.npz', 'wb') as f:
#         np.savez(f, images=images, observations=observations)
