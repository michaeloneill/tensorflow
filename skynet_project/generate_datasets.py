from scipy import misc
import matplotlib.pyplot as plt
import datetime
from dateutil import rrule
import numpy as np
import sqlite3
import os
import csv
from math import cos, sin

plt.rcParams['figure.figsize'] = (3, 6)



def generate_hampstead_observations_csv():

    with open('../data/hampstead_observations/concat_observations.csv', 'wb') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(['Hour', 'Minute', 'Year', 'Month', 'Day', 'WindSpeed', 'MaxGust', 'WindDirection', 'Temperature', 'Humidity', 'Pressure', 'DewPoint', 'RainTotal'])
        with open('../data/hampstead_observations/concat_observations_no_heads.csv', 'rb') as oldfile:
            reader = csv.reader(oldfile)
            writer.writerows([row for row in reader])


def generate_hampstead_weather_db_from_csv():


    conn = sqlite3.connect('../data/hampstead_observations/hampstead.db')
    c = conn.cursor()
    c.execute(''' CREATE TABLE HAMPSTEAD(
    DATE CHAR(10),
    DAYINYEAR REAL,
    HOUR REAL,
    MINUTE REAL,
    WINDSPEED REAL,
    MAXGUST REAL,
    WINDDIRECTIONX REAL,
    WINDDIRECTIONY REAL,
    TEMPERATURE REAL,
    HUMIDITY REAL,
    PRESSURE REAL,
    DEWPOINT REAL,
    RAINTOTAL REAL)''');

    with open('../data/hampstead_observations/concat_observations.csv') as f:
        dr = csv.DictReader(f)
        to_db = []
        errors = {'ValueError':0, 'AttributeError':0, 'Other':0}

        for i in dr:
            try:
                to_db.append(
                    ('{}-{}-{}'.format(i['Year'], i['Month'].zfill(2), i['Day'].zfill(2)),
                     float((datetime.datetime(int(i['Year']), int(i['Month']), int(i['Day']))-datetime.datetime(int(i['Year']), 1, 1)).days), # [0,364]
                     float(i['Hour']),
                     float(i['Minute']),
                     float(i['WindSpeed']),
                     float(i['MaxGust']),
                     cos(float(i['WindDirection'])),
                     sin(float(i['WindDirection'])),
                     float(i['Temperature']),
                     float(i['Humidity']),
                     float(i['Pressure']),
                     float(i['DewPoint']),
                     float(i['RainTotal'])
                ))
            except AttributeError:
                print 'Attribute error raised on row: \n {}'.format(i)
                errors['AttributeError']+=1
                continue
            except ValueError:
                print 'Value Error raised on row: \n {}'.format(i)
                errors['ValueError']+=1
            except:
                print 'Unknown error raised on row: \n {}'.format(i)
                errors['Other']+=1

        print len(to_db)
        print errors

            
    c.executemany('''INSERT INTO HAMPSTEAD (
    DATE,
    DAYINYEAR,
    HOUR,
    MINUTE,
    WINDSPEED,
    MAXGUST,
    WINDDIRECTIONX,
    WINDDIRECTIONY,
    TEMPERATURE,
    HUMIDITY,
    PRESSURE,
    DEWPOINT,
    RAINTOTAL)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''', to_db)

    c.execute('CREATE INDEX date on HAMPSTEAD(DATE)') 

    conn.commit()
    conn.close()

    

def generate_met_weather_db_from_csv():

    conn = sqlite3.connect('../data/met_office_observations/weather.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE WEATHER(
    ID INT PRIMARY KEY,
    FORECASTSITECODE INT,
    OBSERVATIONTIME INT,
    OBSERVATIONDATE CHAR(19),
    WINDDIRECTION INT,
    WINDSPEED INT,
    WINDGUST INT,
    VIZIBILITY INT,
    SCREENTEMPERATURE REAL,
    PRESSURE INT,
    PRESSURETENDENCY CHAR(1),
    SIGNIFICANTWEATHERCODE INT,
    SITENAME TEXT,
    LATITUDE REAL,
    LONGITUDE REAL,
    REGION TEXT,
    COUNTRY TEXT,
    CONTINENT TEXT)''');

    
    with open('../data/met_office_observations/UK_Met_Office_Weather_Open_Data-Observation.csv') as f:
        dr = csv.DictReader(f)
        to_db = [(
            i['ID'],
            i['ForecastSiteCode'],
            i['ObservationTime'],
            i['ObservationDate'],
            i['WindDirection'],
            i['WindSpeed'],
            i['WindGust'],
            i['Visibility'],
            i['ScreenTemperature'],
            i['Pressure'],
            i['PressureTendency'],
            i['SignificantWeatherCode'],
            i['SiteName'],
            i['Latitude'],
            i['Longitude'],
            i['Region'],
            i['Country'],
            i['Continent'])
                 for i in dr]
        
    c.executemany('''INSERT INTO WEATHER (
    ID,    
    FORECASTSITECODE,
    OBSERVATIONTIME,
    OBSERVATIONDATE,
    WINDDIRECTION,
    WINDSPEED,
    WINDGUST,
    VIZIBILITY,
    SCREENTEMPERATURE,
    PRESSURE,
    PRESSURETENDENCY,
    SIGNIFICANTWEATHERCODE,
    SITENAME,
    LATITUDE,
    LONGITUDE,
    REGION,
    COUNTRY,
    CONTINENT) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''', to_db)

    c.execute('CREATE INDEX date on WEATHER(OBSERVATIONDATE)')
              
    conn.commit()
    conn.close()
    
        

def format_time(t):
    return "{:02d}{:02d}".format(t.hour, t.minute)

def format_date(d):
    return "{}{:02d}{:02d}".format(d.year, d.month, d.day)

def time_to_ij(t):
    delta = t - datetime.datetime(100,1,1,0,0)
    s = delta.seconds/60./60./0.5
    i = int(s % 6)
    j = int(s) / 6    
    return i,j

def crop_square_of_sky(image):
    x = 42
    y = 0
    width = 30
    height = 30
    return image[y:(y+height),x:(x+width)]
                    

def get_image_at_datetime(dt):
    base_path = "../data/images/images/raw"
    filename = "{}/{}.jpg".format(base_path, format_date(dt))
    im = misc.imread(filename)
    i,j = time_to_ij(dt)
    sub_img = im[131*j:(102+j*131),i*145:(139+(i)*145),:]
    sub_img = crop_square_of_sky(sub_img)
    if sub_img.shape[1]==0:
        print i, j, dt
    return sub_img

def plot_image_at_datetime(dt):
    sub_img = get_image_at_datetime(dt)
    plt.imshow(sub_img)
    plt.title(dt)
    plt.axis('off')
    plt.show()

def daterange(start_date, end_date):
    # end_date should be one past last date to include
    for n in range(int((end_date-start_date).days)):
        yield start_date + datetime.timedelta(n)
        

def generate_images_targets(start_date, end_date):

    print 'generating targets...'
    targets, remove_dates = generate_targets(start_date, end_date)
    print 'generating images...'
    images = generate_images(start_date, end_date, remove_dates)

    with open('../data/images_targets.npz', 'wb') as f:
        np.savez(f, images=images, targets=targets)
    f.close()
    


def generate_targets(start_date, end_date):
    # end_date should be one past last date to include

    conn = sqlite3.connect('../data/met_office_observations/weather.db')
    c = conn.cursor()
    
    print 'number of records in date range pre removal is {}'.format(int((end_date-start_date).days)*24) # pre date removal

    remove_dates = []
    for date in daterange(start_date, end_date):
        c.execute('''SELECT OBSERVATIONTIME FROM WEATHER
        WHERE FORECASTSITECODE==3672
        AND OBSERVATIONDATE==?
        ''', (date.strftime('%Y-%m-%dT00:00:00'),))
        all_rows = c.fetchall()
        if len(all_rows)!=24:
            print 'found date with {} records (should be 24)'.format(len(all_rows))
            remove_dates.append(date)

    # remove dates corresponding to inconsistently sized images
    remove_dates.append(datetime.datetime(2012,7,22))
    remove_dates.append(datetime.datetime(2012,8,3))

    print 'removing {} dates'.format(len(remove_dates))

    sql = '''SELECT VIZIBILITY FROM WEATHER
    WHERE FORECASTSITECODE==3672
    AND OBSERVATIONDATE NOT IN ({seq})
    AND OBSERVATIONDATE BETWEEN ? AND ?
    ORDER BY OBSERVATIONDATE, OBSERVATIONTIME
    '''.format(seq=','.join(['?']*len(remove_dates)))

    # use remove_dates to store other bindings

    bindings = [d.strftime('%Y-%m-%dT00:00:00') for d in remove_dates+[start_date, end_date-datetime.timedelta(1)]]
    c.execute(sql, bindings)

    all_rows = c.fetchall()
    targets = np.array([float(row[0]) if isinstance(row[0], int) else 0 for row in all_rows])

    print 'number of targets is {}'.format(targets.shape[0])

    targets = (targets-targets.min())/float(targets.max()-targets.min())

    conn.commit()
    conn.close()

    return targets, remove_dates
        


def generate_images(start_date, end_date, remove_dates, start_time=0, end_time=24):
    # end_date and end_time should be one past last date/time to include

    dates_pre = int((end_date-start_date).days) # pre date removal
    print 'number of dates in date range pre removal is {}'.format(dates_pre)

    imHeight = 30
    imWidth = 30
    nChannels = 3

    temp_sz = dates_pre*(end_time-start_time)
    images = np.zeros((temp_sz, imHeight, imWidth, nChannels))

    for i, date in enumerate(daterange(start_date, end_date)):
        if date in remove_dates:
            continue
        else:
            year = date.year
            month = date.month
            day = date.day
            for j in range(end_time-start_time):
                images[i*(end_time-start_time) + j, :, :, :] = get_image_at_datetime(datetime.datetime(year,month,day,hour=j+start_time,minute=0))


    # delete subarrays corresponding to removed dates
    images = images[~np.all(images==0, axis=(1,2,3))]
    print 'number of images is {}'.format(images.shape[0])

    # unit scale

    images = (images-np.amin(images))/(np.amax(images)-np.amin(images)).astype(np.float)

    # # check last image
    # plt.imshow(X[nImages-1, :, :, :])
    # plt.axis('off')
    # plt.show()

    return images



def generate_rnn_trial_dataset(start_date, end_date):

    conn = sqlite3.connect('../data/met_office_observations/weather.db')
    c = conn.cursor()

    print 'number of dates in range pre removal is {}'.format(int((end_date-start_date).days))

    remove_dates = []
    for date in daterange(start_date, end_date):
        c.execute('''SELECT OBSERVATIONTIME FROM WEATHER
        WHERE FORECASTSITECODE==3672
        AND OBSERVATIONDATE==?
        ''', (date.strftime('%Y-%m-%dT00:00:00'),))
        all_rows = c.fetchall()
        if len(all_rows)!=24:
            print 'found date with {} records (should be 24)'.format(len(all_rows))
            remove_dates.append(date)

    print 'removing {} dates'.format(len(remove_dates))

    
    sql = '''SELECT 
    OBSERVATIONTIME, 
    WINDDIRECTION,
    WINDSPEED,
    WINDGUST,
    VIZIBILITY,
    SCREENTEMPERATURE,
    PRESSURE
    FROM WEATHER
    WHERE FORECASTSITECODE==3672
    AND OBSERVATIONDATE NOT IN ({seq})
    AND OBSERVATIONDATE BETWEEN ? AND ?
    AND OBSERVATIONTIME BETWEEN ? AND ?
    ORDER BY OBSERVATIONDATE, OBSERVATIONTIME
    '''.format(seq=','.join(['?']*len(remove_dates)))

    bindings = [d.strftime('%Y-%m-%dT00:00:00') for d in remove_dates+[start_date, end_date-datetime.timedelta(1)]]+[9,17]

    c.execute(sql, bindings)

    all_rows = c.fetchall()

    records = np.array([[float(row[i]) if (isinstance(row[i], int) or isinstance(row[i], float)) else 0 for i in range(len(row))] for row in all_rows])


    # normalise each field independently

    maxes = np.amax(records, axis=0)
    mins = np.amin(records, axis=0)
    records = (records-mins)/(maxes-mins).astype(float)
    
    data = records.reshape([-1, 9, 7])

    observations = data[:, 0:8, :]
    targets = data[:, 1:9, :]
    
    print 'dimension of observations and targets is {} and {} respectively'.format(observations.shape, targets.shape)

    assert np.all(observations[:, 1:, :]==targets[:,:-1, :])

    with open('../data/rnn_trial_dataset.npz', 'wb') as f:
        np.savez(f, observations=observations, targets=targets)
    f.close()


def generate_skynet_trial_dataset(start_date, end_date, start_time=0, end_time=24):
    
    conn = sqlite3.connect('../data/met_office_observations/weather.db')
    c = conn.cursor()

    print  'generating observations ...'
    print 'number of dates in range pre removal is {}'.format(int((end_date-start_date).days))

    remove_dates = []
    for date in daterange(start_date, end_date):
        c.execute('''SELECT OBSERVATIONTIME FROM WEATHER
        WHERE FORECASTSITECODE==3672
        AND OBSERVATIONDATE==?
        ''', (date.strftime('%Y-%m-%dT00:00:00'),))
        all_rows = c.fetchall()
        if len(all_rows)!=24:
            print 'found date with {} records (should be 24)'.format(len(all_rows))
            remove_dates.append(date)

    # remove dates corresponding to inconsistently sized images
    remove_dates.append(datetime.datetime(2012,7,22))
    remove_dates.append(datetime.datetime(2012,8,3))

    print 'removing {} dates'.format(len(remove_dates))

    
    sql = '''SELECT 
    OBSERVATIONTIME, 
    WINDDIRECTION,
    WINDSPEED,
    WINDGUST,
    VIZIBILITY,
    SCREENTEMPERATURE,
    PRESSURE
    FROM WEATHER
    WHERE FORECASTSITECODE==3672
    AND OBSERVATIONDATE NOT IN ({seq})
    AND OBSERVATIONDATE BETWEEN ? AND ?
    AND OBSERVATIONTIME BETWEEN ? AND ?
    ORDER BY OBSERVATIONDATE, OBSERVATIONTIME
    '''.format(seq=','.join(['?']*len(remove_dates)))

    bindings = [d.strftime('%Y-%m-%dT00:00:00')
                for d in remove_dates+[start_date, end_date-datetime.timedelta(1)]]+[start_time,end_time-1]

    c.execute(sql, bindings)

    all_rows = c.fetchall()

    observations = np.array([[float(row[i]) if (isinstance(row[i], int) or isinstance(row[i], float)) else 0 for i in range(len(row))] for row in all_rows])


    # normalise each field independently

    maxes = np.amax(observations, axis=0)
    mins = np.amin(observations, axis=0)
    observations = (observations-mins)/(maxes-mins).astype(float)

    print 'generating_images'
    
    images = generate_images(start_date, end_date, remove_dates, start_time, end_time)
    
    print 'shape of images is {}'.format(images.shape)
    print 'shape of observations is {}'.format(observations.shape)

    print 'max and min in images are {} and {}'.format(np.amax(images), np.amin(images))
    print 'max and min in observations are {} and {}'.format(np.amax(observations), np.amin(observations))
    
    with open('../data/skynet_trial_dataset_30x30x3.npz', 'wb') as f:
        np.savez(f, images=images, observations=observations)
    f.close()


def generate_hampstead_dataset(start_date, end_date, start_time=0, end_time=24):
    # end date should be one past last date

    conn = sqlite3.connect('../data/hampstead_observations/hampstead.db')
    c = conn.cursor()

    print 'generating observations...'
    print 'number of dates in range pre removal is {}'.format(int((end_date-start_date).days))

    remove_dates = []

    # remove images with inconsistent shapes
    remove_dates.append(datetime.datetime(2012,7,22)) 
    remove_dates.append(datetime.datetime(2012,8,3))

    # remove dates with scaffolding
    remove_dates.extend(
        [dt for dt in rrule.rrule(rrule.DAILY,
                        dtstart=datetime.datetime(2013,10,3),
                        until=datetime.datetime(2014,10,29))]
    )

    # remove date with part obstruction
    remove_dates.append(datetime.datetime(2013, 12, 19))

    # remove dates with bedroom
    remove_dates.extend(
        [dt for dt in rrule.rrule(rrule.DAILY,
                        dtstart=datetime.datetime(2014,10,27),
                        until=datetime.datetime(2014,10,29))]
    )

    # more bedroom
    remove_dates.append(datetime.datetime(2015,12,31))
    remove_dates.append(datetime.datetime(2016,1,1))

    # remove empty last date
    remove_dates.append(datetime.datetime(2016,8,27))

    # remove date with missing record
    remove_dates.append(datetime.datetime(2013,4,12))

    remove_dates = list(set(remove_dates)) # removes duplicates

    observations = []

    for date in daterange(start_date, end_date):
        done_with_date = False
        if date not in remove_dates:
            sql = ''' SELECT 
            DAYINYEAR,
            HOUR,
            WINDSPEED,
            MAXGUST,
            WINDDIRECTIONX,
            WINDDIRECTIONY,
            TEMPERATURE,
            HUMIDITY,
            PRESSURE,
            DEWPOINT,
            RAINTOTAL
            FROM HAMPSTEAD
            WHERE DATE=?
            AND MINUTE==0
            AND HOUR BETWEEN ? AND ?
            '''
            bindings = [date.strftime('%Y-%m-%d'), start_time, end_time-1]
        
            c.execute(sql, bindings)
        
            all_rows = c.fetchall()
            if len(all_rows)!=end_time-start_time:
                print 'found date with {} records in time range (should be {})'.format(len(all_rows), end_time-start_time)
                remove_dates.append(date)
                continue
            for row in all_rows:
                if done_with_date:
                    print 'hello'
                    break
                for entry in row:
                    try:
                        float(entry)
                    except ValueError, arg:
                        print 'could not convert entry in date {} to float'.format(date.strftime('%Y-%m-%d'))
                        print arg                                                  
                        remove_dates.append(date)
                        done_with_date=True
                        break
                    except:
                        print 'unknown error in date {}'.format(date.strftime('%Y-%m-%d'))
                        remove_dates.append(date)
                        done_with_date=True
                        break
            # accept date
            observations.extend(all_rows)
            assert sorted(remove_dates)==sorted(list(set(remove_dates)))

    print '{} dates removed'.format(len(remove_dates))

    # convert to float
    observations = np.array([[float(entry) for entry in record] for record in observations])

    # normalise each field independently

    maxes = np.amax(observations, axis=0)
    mins = np.amin(observations, axis=0)
    observations = (observations-mins)/(maxes-mins).astype(float)

    print 'generating_images...'

    images = generate_images(start_date, end_date, remove_dates, start_time, end_time)
    
    print 'shape of images is {}'.format(images.shape)
    print 'shape of observations is {}'.format(observations.shape)

    print 'max and min in images are {} and {}'.format(np.amax(images), np.amin(images))
    print 'max and min in observations are {} and {}'.format(np.amax(observations), np.amin(observations))
    
    with open('../data/hampstead_dataset.npz', 'wb') as f:
        np.savez(f, images=images, observations=observations)
    f.close()
    


if __name__ == "__main__":

    # generate_met_weather_db_from_csv()

    start_date = datetime.datetime(2012, 6, 27)
    end_date = datetime.datetime(2016, 8, 28) # one past last date
    start_time = 9
    end_time = 18 # one past last time

    # generate_images_targets(start_date, end_date)
    
    # generate_rnn_trial_dataset(start_date, end_date)

    # generate_skynet_trial_dataset(start_date, end_date, start_time, end_time)

    # generate_hampstead_observations_csv

    # generate_hampstead_weather_db_from_csv()

    generate_hampstead_dataset(start_date, end_date, start_time, end_time)


    
