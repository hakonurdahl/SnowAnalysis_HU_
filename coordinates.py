import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import pandas as pd
from A_funcstat import closest
import requests

# The main output from this file is a points that are known to have SWE values in the SeNorge database
# For automation purposes, an API is used to find appropriate points using a the name of the municipality along with keyword
# If needed, the function can return a variety of interesting results, as a list of points within 
# the square kilometer around the main points, or a list of the points tested for SWE values.

# A function that find the coordinates of a place from a string https://nominatim.openstreetmap.org
def get_coordinates(building_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': building_name,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'reliability_analysis/1.0 (hakon.urdahl@outlook.com)'  # Nominatim requires identifying info
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if data:
        lat = data[0]['lat']
        lon = data[0]['lon']
        return float(lat), float(lon)
    else:
        print("Didn't find coordinates for", building_name)
        return None

def coordinates(name, ds, save_all_attempts=False):

    # Coordinate transformation required due to SeNorge dataset and Nominatim mismatch 
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    to_latlon = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

    # Geographical coordinates from keyword
    # The name of the municipality + one of the key words below are meant to represent a part of the
    # municipality that is relatively populated. 

    keywords = [" rådhus", " barneskole", " skule", " kirke", " skole"]
    
    # Special-case municipalities with manually specified coordinates
    special_coordinates = {
        "Ål": (60.63024, 8.56121),       
        "Meland": (60.51757, 5.23958),
        "Nesset": (62.77575, 8.06621), 
        "Båtsfjord": (70.63459, 29.71046),  
        "Austrheim": (60.76380, 4.91567),
        "Birkenes": (58.33485,8.23298),
        "Dovre": (61.98560, 9.24943),
        "Gjerstad": (58.88114, 9.01837),
        "Hol": (60.61607,8.30206),
        "Lillestrøm": (59.96151, 11.05753),
        "Røyrvik": (64.88776, 13.55853),
        "Valle": (59.21175, 7.5334),
        "Vang": (61.12518, 8.57213),
        "Vik": (61.08815, 6.58583),
        "Vinje": (59.5689, 7.98935),
        "Norddal": (62.30, 7.25),
        "Fjord": (62.29811193390707, 7.245234119948996),
        "Nesna": (66.19917491265197, 13.034484695551873),
        "Gamvik": (71.04147355360465, 27.8681967921552),
        "Hammerfest": (70.6601431489694, 23.697329832073535),
        "Loppa": (70.2325140824077, 22.340843599656),
        "Nesseby": (70.16691702484962, 28.55303337930349)
        }

    # Determine coordinates
    if name in special_coordinates:
        data = special_coordinates[name]
    else:
        for keyword in keywords:
            data = get_coordinates(name + keyword)
            if data is not None:
                break

    latitude, longitude = data
   
    x_, y_ = transformer.transform(longitude, latitude)

    ### Algorithm to make sure the point contain SWE values ###
    ###########################################################

    snow_water_equivalent = ds['snow_water_equivalent__map_rcp85_daily']

    latitudes = ds['Yc'].values
    longitudes = ds['Xc'].values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    distances = np.sqrt((lat_grid - y_)**2 + (lon_grid - x_)**2)
    closest_indices = np.unravel_index(np.argsort(distances, axis=None)[:21], distances.shape)

    # Alternative to check what points were attempted
    if save_all_attempts:
        attempted_points = []
        attempted_points.append((latitude, longitude))

    for j in range(21):
        y_nearest = lat_grid[closest_indices[0][j], closest_indices[1][j]]
        x_nearest = lon_grid[closest_indices[0][j], closest_indices[1][j]]
        swe_at_point = snow_water_equivalent.sel(Yc=y_nearest, Xc=x_nearest, method='nearest')
        swe_array = swe_at_point.values

        # Convert back to lat/lon
        actual_lon, actual_lat = to_latlon.transform(x_nearest, y_nearest)

        if save_all_attempts:
            attempted_points.append((actual_lat, actual_lon))

        # If a points returing SWE values are found, the algorithm ends
        if not np.isnan(swe_array).all():
            actual_lon, actual_lat = to_latlon.transform(x_nearest, y_nearest)      
            if save_all_attempts:
                return attempted_points
            break
    
    ###########################################################

    coordinate_samples=[]
    coordinate_samples.append((float(actual_lat), float(actual_lon)))

    # Function for a evenly spread out grip of points within the square kilometer. 
    # Was first thought of as a means of finding the average elevation for calculations of char.
    # However, the OI scheme include elevation effects. Could still be interesting, but currently not in use.
    grid=0
    if grid ==1:
        coordinate_samples=[]
        test_lat=actual_lat-0.01
        for i in range(20):
            
            test_lon = actual_lon - 0.01        

            for k in range(10):
                
                lat, lon = closest(test_lat, test_lon)
                if lon==actual_lon and lat==actual_lat:
                    coordinate_samples.append((float(test_lat), float(test_lon)))

                test_lon+=0.002
            
            test_lat+=0.001

    return coordinate_samples

#Test
test_run = 0

if test_run==1:
    opendap_url = f'https://thredds.met.no/thredds/dodsC/KSS/Klima_i_Norge_2100/utgave2015/SWE/MPI_RCA/rcp85/rcp85_MPI_RCA_SWE_daily_2025_v4.nc'

    try:
        # Open the dataset    
        ds_ = xr.open_dataset(opendap_url, chunks=None)
        
    except Exception as e:
        print(f"Could not process year 2024: {e}")

    test_mun = "Norddal"
    print(coordinates(test_mun, ds_, save_all_attempts=True))
    #print(test_mun + ',' + '"' + str(coordinates(test_mun, ds_, save_all_attempts=True)) + '"')


