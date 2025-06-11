import csv
import os
import xarray as xr
import numpy as np
import pandas as pd
from coordinates import coordinates
from elevation import get_elevations
from swe import measurements
from FORM import calibration, municipality_form, char, char_T50, prop
from C_Input_AHG import input_data


# Many of the calculations in this project are computationally intensive or involve large datasets. 
# This Python file automates the extraction, computation, and storage of the necessary data.
# The results are written to CSV files for later use in analysis, visualization, and reporting.
# This script stores results in CSV files to avoid recalculating the same values repeatedly.
# Several debugging prints are possible, as well as the oportunity to start at a specific municiplaity for some functions.
# The datasets from SeNorge could be dowloaded in it's entirety, which would have some benefits for efficiency,
# if the SWE values are to be used in different ways. However, only storing maximum values have some benefits
# regarding storage space and a more interactive and dynamic script.

# Folder where all data files are stored
folder_path = "C:/Users/hakon/SnowAnalysis_HU/stored_data"
os.makedirs(folder_path, exist_ok=True)

# Get a list of all municipalities from the EN1991 csv file
mun_csv_path = os.path.join(folder_path, "EN1991.csv")
mun_df = pd.read_csv(mun_csv_path)
municipalities_data = mun_df["municipality"].tolist()

# Generic function to write CSV output
def write_csv(filename, header, rows, mode="w"):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV written: {file_path}")

# Download dataset files sequentially
def load_swe_datasets(years, scenario=None):
    ds_list = []
    for year in range(years[0], years[1]):
        # The historical and future projections are from different URLs
        if scenario:
            url = f'https://thredds.met.no/thredds/dodsC/KSS/Klima_i_Norge_2100/utgave2015/SWE/MPI_RCA/{scenario}/{scenario}_MPI_RCA_SWE_daily_{year}_v4.nc'
        else:
            url = f'https://thredds.met.no/thredds/dodsC/senorge/seNorge_snow/swe/swe_{year}.nc'
        try:
            ds_list.append(xr.open_dataset(url, chunks=None))
        except Exception as e:
            print(f"Failed to load {year}: {e}")
    return ds_list

# Difference between two result CSVs
def print_to_csv_diff(time_1, var_1, time_2, var_2):
    def read_results(var, time):
        path = os.path.join(folder_path, f"{var}_{time}.csv")
        with open(path, mode='r', encoding='utf-8') as f:
            return {row["municipality"]: float(row["var"]) for row in csv.DictReader(f)}

    vals_1 = read_results(var_1, time_1)
    vals_2 = read_results(var_2, time_2)

    results = []
    for m in vals_2:
        if m in vals_1:
            results.append([m, vals_1[m] - vals_2[m]])
        else:
            print(f"Missing in first: {m}")
    write_csv(f"diff_{var_1}_{time_1}_{time_2}.csv", ["municipality", "var"], results)


# Generic template for printing variables
def print_single_value_csv(task_name, compute_func, time=None, start_municipality=None):
    rows = []
    start = start_municipality is None
    for municipality in municipalities_data:
        if not start and municipality == start_municipality:
            start = True
        if start:
            result = compute_func(municipality) if not time else compute_func(municipality, time)
            rows.append([municipality, result])
            print(f"{task_name} added: {municipality}")
    write_csv(f"{task_name}.csv", ["municipality", "var"], rows)

# Specific versions using the generic writer
def print_to_csv_swe(time, start_municipality=None):
    scen = input_data[time]["scenario"]
    years = input_data[time]["period"]
    ds_list = load_swe_datasets(years, scen)
    def swe_func(m, t): 
        swe = measurements(m, ds_list, scen)
        return "[nan]" if np.isnan(swe).all() else repr(swe.tolist())      
    print_single_value_csv(f"swe_{time}", swe_func, time=time, start_municipality=start_municipality)

def print_to_csv_points(start_municipality=None):
    ds = load_swe_datasets((2022,2023), "rcp85")
    def points_func(m): return coordinates(m, ds[0])
    print_single_value_csv("points", points_func, start_municipality=start_municipality)

def print_to_csv_char(start_municipality=None):
    print_single_value_csv("char_ec", char, start_municipality=start_municipality)

def print_to_csv_elevation(start_municipality=None):
    print_single_value_csv("elevation", get_elevations, start_municipality=start_municipality)

def print_to_csv_beta(time, start_municipality=None):
    def beta_func(m, t): return municipality_form(m, t, None)[0]
    print_single_value_csv(f"beta_{time}", beta_func, time=time, start_municipality=start_municipality)

def print_to_csv_cov(time, start_municipality=None):
    def cov_func(m, t): return prop(m, t)[1]
    print_single_value_csv(f"cov_{time}", cov_func, time=time, start_municipality=start_municipality)

def print_to_csv_T50_char(start_municipality=None):
    print_single_value_csv(f"T50_char_tot", char_T50, start_municipality=start_municipality)

def print_to_csv_T50_beta(start_municipality=None):
    c_df = pd.read_csv(os.path.join(folder_path, "T50_char_tot.csv")).set_index("municipality")
    def beta_func(m): 
        c = c_df.loc[m, "var"]
        return municipality_form(m, "tot", c)[0] 
    print_single_value_csv(f"T50_beta_tot", beta_func, start_municipality=start_municipality)

def print_to_csv_char_opt(time, start_municipality=None):
    def opt_func(m,t): return calibration(m,t,3.8)[0]
    print_single_value_csv(f"opt_char_{time}", opt_func, time=time, start_municipality=start_municipality)

def print_to_csv_beta_opt(time, start_municipality=None):
    c_df = pd.read_csv(os.path.join(folder_path, f"opt_char_{time}.csv")).set_index("municipality")
    def beta_func(m, t): 
        c = c_df.loc[m, "var"]
        return municipality_form(m, t, c)[0] 
    print_single_value_csv(f"opt_beta_{time}", beta_func, time=time, start_municipality=start_municipality)


# Example usage
#print_to_csv_beta("tot")
