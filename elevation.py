import requests
import pyproj
from pyproj import Transformer
import ast
import pandas as pd
import csv
import os


# This file returns elevation for a points by utilizing an API from hoydedata.no
# DTM is preferred, but if no DTM is available, the DOM is returned, the differnce beeing that
# the former is terrein relative to ocean level, while DOM includes vegetation and such

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

def get_dom_elevation(easting, northing):
    """Fetch elevation using WMS GetFeatureInfo from DOM."""
    wms_url = "https://wms.geonorge.no/skwms1/wms.hoyde-dom-prosjekt-punkttetthet"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": "dom_punkttetthet",
        "QUERY_LAYERS": "dom_punkttetthet",
        "CRS": "EPSG:25833",
        "INFO_FORMAT": "text/plain",
        "WIDTH": 1,
        "HEIGHT": 1,
        "I": 0,
        "J": 0,
        "BBOX": f"{easting},{northing},{easting + 0.01},{northing + 0.01}"
    }

    try:
        response = requests.get(wms_url, params=params)
        response.raise_for_status()
        text = response.text
        if "value" in text.lower():
            for line in text.splitlines():
                if "value" in line.lower():
                    return float(line.split('=')[-1].strip())
    except Exception as e:
        print(f"DOM fallback failed: {e}")
    return None

def get_elevations(name):

    # Get coordinates from stored file
    folder_path = folder_path = "C:/Users/hakon/SnowAnalysis_HU/stored_data"
    coordinates_df = pd.read_csv(os.path.join(folder_path, "points.csv")).set_index("municipality")
    coordinates_str = coordinates_df.loc[name, "var"]
    coordinates = ast.literal_eval(coordinates_str)

    # Try DTM, then fallback to DOM if DTM returns None
    elevations = []

    for lat, lon in coordinates:
        
        # Mismatch between csv file format and API requires coordinate transformation
        easting, northing = transformer.transform(lon, lat)
        # First attempt: DTM
        url = f"https://ws.geonorge.no/hoydedata/v1/punkt?nord={northing}&ost={easting}&koordsys=25833&geojson=false"
        try:
            response = requests.get(url)
            response.raise_for_status()
            elevation_data = response.json()

            if "punkter" in elevation_data and len(elevation_data["punkter"]) > 0:
                elevation = elevation_data["punkter"][0]["z"]
                elevations.append(elevation)
                continue  # skip DOM fallback if DTM worked
        except Exception as e:
            print(f"DTM fetch failed for ({lat}, {lon}): {e}")

        # Fallback to DOM via WMS
        elevation = get_dom_elevation(easting, northing)
        elevations.append(elevation)

    return elevations

# Example run
run = 0
if run == 1:
    results = get_elevations("Oslo")
    print(results)
