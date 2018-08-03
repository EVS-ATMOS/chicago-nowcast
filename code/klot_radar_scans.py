# Jason Hemedinger
# Argonne National Laboratory
#SULI Summer 2018
#Project titled: Storm Cell Tracking and Nowcasting for Argonne National Laboratory


from pylab import *
import pyart, boto3, tempfile, os, shutil, datetime, matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import animation
from botocore.handlers import disable_signing
from tint import Cell_tracks
from tint import animate as tint_animate
from tint.visualization import embed_mp4_as_gif
from math import sin, cos, sqrt, atan2, radians
from glob import glob


def get_radar_scan(station='KLOT', date=None, key_index=-15):
    '''
    Function will pull the latest radar scan from any radar site using 
    Amazon S3.
    ----------
    Station = Four letter NEXRAD identifier
              Example: 'KEPZ'
    Date = default is none for current date, else enter date in format "YYYY/MM/DD"
    Ex: date ='2013/11/17
    Key_index = Number of keys you want pulled from most recent scan.
    Ex: key_index = -15 would pull ht most recent 15 scans
    '''
    
    #creating a bucket and a client to be able to pull data from AWS and setting 
    #it as unsigned
    bucket = 'noaa-nexrad-level2'
    s3 = boto3.resource('s3')
    s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    
    #connects the bucket create above with radar data
    aws_radar = s3.Bucket(bucket)
    
    #setting the date and time to current.
    #this will allow for allow the current date's radar scands to be pulled
    if date == None:
        target_string = datetime.datetime.utcnow().strftime('%Y/%m/%d/'+station)
    else:
        target_string = date+'/'+station
        
    
    for obj in aws_radar.objects.filter(Prefix= target_string):
        '{0}:{1}'.format(aws_radar.name, obj.key)
    my_list_of_keys = [this_object.key for this_object in aws_radar.objects.filter(Prefix= target_string)]
    keys = my_list_of_keys[key_index:]
    print(keys)
    return aws_radar, keys


def new_directory(date = 'current', 
                  year = datetime.datetime.utcnow().strftime('%Y'), 
                  month = datetime.datetime.utcnow().strftime('%m'), 
                  day = datetime.datetime.utcnow().strftime('%d'), 
                  hour = datetime.datetime.utcnow().strftime('%H'), 
                  path = '/home/jhemedinger/suli_projects/chicago-nowcast/events'):
    """ 
    Function will create a new directory and save all data and images to that file
    ----------
    date: options are either current or past- current will create a file of the current date and time
          past will allow for a file of a previous date and time to be created
          Ex: date='past'
    year: year of the date for the file being created. If no yearis given then current year is used
    month: same as year but for month
    day: same as year and month but for day
    hour: hour for which the data is from, if not set hour is set to current
    paht: path for where the new directory will be created and saved
    """
    
    if date == 'past':
        past_date = str(datetime.datetime(year, month, day).strftime('%Y_%m_%d'))
        out_dir_path = path+'/'+past_date
        event_date = str(datetime.datetime(year, month, day, hour).strftime('%Y%m%d-%H'))
    elif date == 'current':
        cur_date = str(datetime.datetime.utcnow().strftime('%Y_%m_%d'))
        out_dir_path = path+'/'+cur_date
        event_date = str(datetime.datetime.utcnow().strftime('%Y%m%d-%H'))
    out_dir = os.makedirs(out_dir_path, exist_ok=True)    
    out_path_dir = out_dir_path+'/'+event_date+'Z'
    out_path = os.makedirs(out_path_dir, exist_ok=True)
    print('current saving directory:', out_path_dir)
    return out_path_dir


#setting the radar information to be pulled from AWS as well as created a new directory for the data to be saved to
aws_radar, keys = get_radar_scan()
out_path_dir = new_directory()


#creating a radar animation using pyart and matplotlib functions
def animate(nframe):
    plt.clf()
    localfile = tempfile.NamedTemporaryFile()
    aws_radar.download_file(keys[nframe], localfile.name)
    radar = pyart.io.read(localfile.name)
    display = pyart.graph.RadarMapDisplay(radar)
    # Delete radar after use to save memory.
    del radar
    display.plot_ppi_map('reflectivity', sweep=0, resolution='l',
                         vmin=-8, vmax=64, mask_outside=False, 
                         fig=fig, width=350000, height=350000, 
                         cmap = pyart.graph.cm.LangRainbow12 )

    display.basemap.drawcounties()
    display.plot_point(-87.981810, 41.713969 , label_text='ANL', symbol='ko')
fig = plt.figure(figsize=(10, 8))

anim_klot = animation.FuncAnimation(fig, animate,
                                    frames=len(keys))
anim_klot.save(out_path_dir + '/reflectivity_animation.gif', 
               writer='imagemagick', fps=2)
plt.show()
plt.close()


#turing the data into grid data and saving it to a folder
def get_grid(aws_radar, key):
    localfile = tempfile.NamedTemporaryFile()
    aws_radar.download_file(key, localfile.name)
    radar = pyart.io.read(localfile.name)
    grid = pyart.map.grid_from_radars(
            radar, grid_shape=(31, 401, 401),
            grid_limits=((0, 15000), (-200000, 200000), (-200000, 200000)),
            fields=['reflectivity'], weighting_function='Barnes', gridding_algo='map_gates_to_grid',
            h_factor=0., nb=0.6, bsp=1., min_radius=200.)
    return grid

for num,key in enumerate(keys):
    print('saving grid', num)
    grid = get_grid(aws_radar, key)
    name = os.path.join(out_path_dir, 'grid_' + str(num).zfill(3) + '.nc')
    pyart.io.write_grid(name, grid)
    del grid


#reading in the gridded data to be used with TINT
files = glob(out_path_dir + '/grid_*')
files.sort()


#creating a grid generator to be able to read the grids into TINT
grid_gen = (pyart.io.read_grid(f) for f in files)


#creating the cell tracks and changing the minimum threshold value for reflectivity
tracks_obj = Cell_tracks()
tracks_obj.params['FIELD_THRESH']=35

tracks_obj.get_tracks(grid_gen)

tracks_obj.tracks


# this section is only necessary to run if there is already a file within the directory with the same name
#if you rerun code without deleting the old file an error will occur since the old file was not overwritten
if os.path.exists(out_path_dir + '/tracks_animation.mp4'):
    print(out_path_dir + '/tracks_animation.mp4'
          + ' already exists, removing file')
    os.remove(out_path_dir + '/tracks_animation.mp4')

#using the animate function within TINT to get the cell tracks
grid_gen = (pyart.io.read_grid(f) for f in files)
tint_animate(tracks_obj, grid_gen, os.path.join(out_path_dir, 'tracks_animation'), tracers=True, 
             cmap=pyart.graph.cm.LangRainbow12)#, lat_lines=lat_lines, lon_lines=lon_lines)


embed_mp4_as_gif(out_path_dir + '/tracks_animation.mp4')


#seperating the data by uid
cells = tracks_obj.tracks.groupby(level='uid')
for uid in cells:
    print(uid)


tracks_obj.tracks.groupby(level='uid').size().sort_values(ascending=False)[:]


#pulling the data for a specific uid
df_0 = pd.DataFrame(tracks_obj.tracks.xs('5', level='uid'))
lons, lats = np.array(df_0['lon']), np.array(df_0['lat'])
time = np.array(pd.to_datetime(df_0['time']))
print(df_0)


#creating the linear regression using polyfit and poly1d
fit = polyfit(lons[:10],lats[:10],1)
fit_fn = poly1d(fit)


#plotting the regression and the lat/lon data and showing the 95% confidence interval of the regression model
fig = plt.figure(figsize=(10,8))
plt.plot(lons[:10], lats[:10], '--ok', label='Latitude/Longitude')
sns.regplot(lons[:10], lats[:10], color='b')
#for i, txt in enumerate(time[:11]):
#    plt.annotate(txt, (lons[:][i], lats[:][i]))
plt.plot(lons[:10], fit_fn(lons[:10]), '-b',
         label='Linear Regression \nwith 95% Confidence Interval')

plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.legend(loc=4)
#font = { 'family' : 'normal',
#            'size'   : 15}
#matplotlib.rc('font', **font)
#plt.grid()
plt.title('June 26, 2018 FIELD_THRESH=35dBz')
plt. savefig(out_path_dir + '/regression.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


#plotting a time series of the latitude and longitude data
t = (time[:10] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
x, y = lats[:10], lons[:10]

fit_lat = polyfit(t,x,1)
fit_lon = polyfit(t,y,1)
fit_fn_lon = poly1d(fit_lon)
fit_fn_lat = poly1d(fit_lat)
#font = { 'family' : 'normal',
#            'size'   : 15}
#matplotlib.rc('font', **font)
fig = plt.figure(figsize=(10,8))
plt.plot(time[:10], x, 'ro', time[:10], fit_fn_lat(t), '--k')

plt.xlabel('TIME (UTC)')
plt.ylabel('LATITUDE')
plt.title('Latitudinal Time Series')

plt.savefig(out_path_dir + '/lat_reg.png', dpi=300)
plt.show()
plt.close()

fig = plt.figure(figsize=(10,8))
plt.plot(time[:10], y, 'bo', time[:10], fit_fn_lon(t), '--k')
plt.xlabel('TIME (UTC)')
plt.ylabel('LONGITUDE')
plt.title('Longitudinal Time Series')
plt.savefig(out_path_dir + '/lon_reg.png', dpi=300)
plt.show()
plt.close()


def lats_lons(minimum, maximum, interval):
    '''
    Will predict lat/lon for a given time interval.
    Returns time, lat, and lon
    beginning: beginning of the time interval
    end: end of interval
    interval: time interval in minutes
    
    Ex: lats_lons(10, 70, 10) will find the lat/lon 
    for the next hour every 10 minutes.
    '''
    minimum = minimum
    maximum = maximum
    interval = interval
    arr = np.arange(minimum, maximum, interval) 
    my_time = []
    for i in arr:
        my_time.append(time[:10][-1] + np.timedelta64(str(i), 'm'))
    my_new_time = np.array(my_time)
    nts = ((my_new_time - np.datetime64('1970-01-01T00:00:00Z')) 
           / np.timedelta64(1, 's'))
    my_new_lat = fit_fn_lat(nts)
    my_new_lon = fit_fn_lon(nts)
#    print(my_new_time)
#    print(my_new_lon)
#    print(my_new_lat)

    return my_new_time, my_new_lat, my_new_lon

#getting future lat/lon points
my_new_time, my_new_lat, my_new_lon = lats_lons(10,90,10)


#calculating the distance the center of a cell is from Argonne using the Haversine formula
#unit for distance is km
for i in range(8):
    anl_lon = radians(-87.981810)
    anl_lat = radians(41.713969)
    storm_lon = radians(my_new_lon[i])
    storm_lat = radians(my_new_lat[i])
    pre_time = (my_new_time[i])
    
    dlon = storm_lon - anl_lon
    dlat = storm_lat - anl_lat
    
    a = sin(dlat / 2)**2 + cos(anl_lat) * cos(storm_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    R = 6373.0
    distance = R * c
#    print(distance)

    
#setting distance(s) to determine of ANL will be hit by a strom cell
#distance of 12km was used because the average radius of a storm is 12km
    if distance <= 12:
        print('At', pre_time, 'storm is forecsted to be at ANL')    


#animating using matplotlib and pyart
def animate(nframe):
    plt.clf()
    localfile = tempfile.NamedTemporaryFile()
    aws_radar.download_file(keys[11:20][nframe], localfile.name)
    radar = pyart.io.read(localfile.name)
    display = pyart.graph.RadarMapDisplay(radar)
    # Delete radar after use to save memory.
    del radar
    display.plot_ppi_map('reflectivity', sweep=0, resolution='l',
                         vmin=-8, vmax=64, mask_outside=True, 
                         fig=fig, width=85000, height=85000,
                         cmap=pyart.graph.cm.LangRainbow12)
    display.basemap.drawcounties()
    display.plot_line_geo(lons[-8:][:nframe], lats[-8:][:nframe], '-k', label='Observed storm path')
    display.plot_line_geo(my_new_lon, my_new_lat, '--r', label='Forecasted storm path')
    display.plot_point(-87.981810, 41.713969 , label_text='ANL', symbol='k*', label_offset=(-0.04,0.01))
    plt.legend(loc=3)
fig = plt.figure(figsize=(12, 8))
#font = { 'family' : 'normal',
#            'size'   : 15 }
anim_klot = animation.FuncAnimation(fig, animate, 
                                    frames=len(keys[11:20]))
anim_klot.save(out_path_dir + '/ref_track_animation_test.gif', 
               writer='imagemagick', fps=1)

plt.show()
plt.close()

#creating ppi image of last scan and plotting predicted and observed storm path 
localfile = tempfile.NamedTemporaryFile()

fig=plt.figure(figsize=(12,8))

aws_radar.download_file(keys[-1], localfile.name)
radar = pyart.io.read(localfile.name)
display = pyart.graph.RadarMapDisplay(radar)
#font = { 'family' : 'normal',
#            'size'   : 15 }
#matplotlib.rc('font', **font)
display.plot_ppi_map('reflectivity', sweep=0, resolution='l', 
                     vmin=-8, vmax=64, mask_outside=True, 
                     width=90000, height=90000, 
                     cmap=pyart.graph.cm.LangRainbow12)
display.basemap.drawcounties()
display.plot_line_geo(lons[-8:], lats[-8:], '-k', label='Observed Storm Path')
display.plot_line_geo(my_new_lon, my_new_lat, '--r', label='Forecasted Storm Path')
display.plot_point(-87.981810, 41.713969 , label_text='ANL', symbol='k*', label_offset=(-0.04, 0.01))
plt.legend(loc=4)

plt.savefig(out_path_dir +'/reg_plot_radar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
