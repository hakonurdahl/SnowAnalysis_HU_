import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import B_mainclass as form
import C_Input_AHG as inp
import D_Preprocessing as prep
from statistics import mean
from math import ceil
from scipy.stats import gumbel_r
import csv
import os
from ast import literal_eval
import pandas as pd

# The main function is to calculate reliability indices for municipalities given SWE data input. 
# In addition a reliaiblity-based calibration of the optimal characteristic value is included.
# Other results as the 50-year return period characteristic values and parameters can also be calculated here.

folder_path = "C:/Users/hakon/SnowAnalysis_HU/stored_data"



# Lazy-loaded SWE cache to avoid reloading per function call
swe_data_cache = {}

def get_swe_data(time):
    if time in swe_data_cache:
        return swe_data_cache[time]

    path = os.path.join(folder_path, f"swe_{time}.csv")
    swe_df = pd.read_csv(path)

    # Convert stringified lists to actual lists of floats
    swe_df["swe"] = swe_df["var"].apply(lambda x: [float(i) for i in literal_eval(x)])

    # Create dictionary: {municipality: list of floats}
    swe_data = dict(zip(swe_df["municipality"], swe_df["swe"]))

    swe_data_cache[time] = swe_data
    return swe_data




# Calculate characteristic snow load (NS-EN1991-1-3)
def char(name):
    # Load NS-EN1991-1-3 NA data
    en1991_df = pd.read_csv(os.path.join(folder_path, "EN1991.csv")).set_index("municipality")

    # Load elevation data
    elevation_df = pd.read_csv(os.path.join(folder_path, "elevation.csv")).set_index("municipality")
    elevation_df["var"] = elevation_df["var"].apply(lambda x: mean(literal_eval(x)))
    m = en1991_df.loc[name]
    elevation = elevation_df.loc[name, "var"]
    n = max(ceil((elevation - m["hg"]) / 100), 0)
    sk = m["sk_0"] + m["dsk"] * n
    return min(sk, m["sk_maks"])

# Calculate characteristic snow load based on T=50 year return period
def char_T50(name):
    snow_maxima = get_swe_data("tot")[name]
    loc, scale = gumbel_r.fit(snow_maxima)
    x_k = gumbel_r.ppf(0.98, loc=loc, scale=scale)
    return x_k * 9.8 * 2 / 1000  # Converting mm SWE to kN/m

# Get mean and CoV of snow load for given municipality and time
def prop(name, time):
    snow_maxima = get_swe_data(time)[name]

    if np.sum(snow_maxima) < 10:
        loc, scale = 0.01, 0.01
    else:
        loc, scale = gumbel_r.fit(snow_maxima)

    gamma = 0.57722
    mean_gumbel = loc + gamma * scale
    std_gumbel = (np.pi / np.sqrt(6)) * scale
    mean_snow = mean_gumbel * 9.8 * 2 / 1000    # Converting mm SWE to kN/m
    cov_snow = std_gumbel / mean_gumbel

    return mean_snow, cov_snow

# Calculate beta (reliability index), optionally with custom characteristic value
def municipality_form(name, time, char_assigned=None):
    mean_snow, cov_snow = prop(name, time)
    char_val = char_assigned if char_assigned is not None else char(name) * 2 # *2 from roof-width 

    X = prep.RandomVariablesAux(mean_snow, cov_snow, char_val)
    g_ = inp.StartValues()
    deq = 1

    P_ = X['Y32']
    XX_ = X['Y11']
    Q_ = X['Z2']
    XQ_ = X[X['Z2']['MUV']]
    XR_ = X['X11']
    R_ = X[XR_['RV']]
    G_ = X[XR_['GV']]

    zet = form.ZBETA(XR=XR_, R=R_, XX=XX_, G=G_, P=P_, XQ=XQ_, Q=Q_, g=g_, d=deq)
    z = zet.__zeta__()
    BETA, ALPHA = zet.f1(z)

    #BETA = 1
    if BETA == 1:
        BETA_mcs = zet.mcstest(z)
        return (12 if BETA_mcs > 100 else BETA_mcs), ALPHA

    return BETA, ALPHA

# Used to find the so-called optimal characteristic value, not bound by return period
def calibration(name, time, beta_target):
    beta_target_range = (beta_target - 0.01, beta_target + 0.01)

    def func_opt(char_opt):
        beta, _ = municipality_form(name, time, char_opt)
        return abs(beta - np.mean(beta_target_range))

    bounds_list = [(0, 5.1), (5, 10.1), (10, 18.1), (18, 27), (26, 40)]

    for bounds in bounds_list:
        res = opt.minimize_scalar(func_opt, bounds=bounds, method='bounded')
        optimal_char = res.x
        optimal_beta, _ = municipality_form(name, time, optimal_char)

        if beta_target_range[0] <= optimal_beta <= beta_target_range[1]:
            return optimal_char, optimal_beta

    return "Error", "Error"

run=0

if run ==1:

    municiaplities=["Nærøysund", "Evenes", "Skjervøy", "Jølster", "Nordkapp"]

    for municipality in municiaplities:
        
        BETA_, ALPHA_=municipality_form(municipality, "tot")
        print(municipality,",",BETA_)

        #print(calibration("Oslo", "tot", 3.8))
        #print(char_T50("Oslo"))

