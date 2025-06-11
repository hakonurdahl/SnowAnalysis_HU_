import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gumbel_r
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
})

fontsize_ = 22


# Two plots used in the Methods chapter and Theory chapter, respectivly. The first show the process
# of collecting SWE values, choosing the yearly extremes and using these to estimate a Gumbel distribution.
# The second illustrate the proess of bias correction.

# File paths and station name 
csv_file = '/Users/hakon/SnowAnalysis_HU/stored_data/swe_Gloshaugen.csv'
output_folder = '/Users/hakon/SnowAnalysis_HU/Figures/main_output/'
station_name = os.path.splitext(os.path.basename(csv_file))[0]


def plot_combined_timeseries_and_percentile():
    # Plot SWE time series and percentile plot 
    data = pd.read_csv(csv_file)
    data['time'] = pd.to_datetime(data['time'])
    data['water_year'] = data['time'].apply(lambda x: x.year + 1 if x.month >= 7 else x.year)
    data = data[(data['time'] >= '1958-07-01') & (data['time'] <= '2023-06-30')]

    max_swe_per_year = data.loc[data.groupby('water_year')['swe'].idxmax()].reset_index(drop=True)

    first_year = data['water_year'].min()
    if data[data['water_year'] == first_year]['time'].min().month > 7:
        max_swe_per_year = max_swe_per_year[max_swe_per_year['water_year'] != first_year]

    swe_values = max_swe_per_year['swe'].values
    n = len(swe_values)
    percentiles = np.arange(1, n + 1) / (n + 1) * 100
    sorted_indices = np.argsort(swe_values)
    sorted_swe_values = swe_values[sorted_indices]

    loc, scale = gumbel_r.fit(swe_values)
    x_fit = np.linspace(0, sorted_swe_values.max(), 500)
    cdf_gumbel = gumbel_r.cdf(x_fit, loc=loc, scale=scale) * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Time series subplot
    axs[0].plot(data['time'], data['swe'], label='Daily SWE', color='skyblue')
    axs[0].scatter(max_swe_per_year['time'], max_swe_per_year['swe'], color='blue', s=30, label='Yearly Maxima')
    axs[0].set_xlabel('Date', fontsize=fontsize_-3)
    axs[0].set_ylabel('SWE (mm)', fontsize=fontsize_-3)
    axs[0].set_title(r'SWE Time Series - Trondheim, Gl\o shaugen', fontsize=fontsize_)
    axs[0].tick_params(axis='both', labelsize=fontsize_-4)
    axs[0].legend(fontsize=fontsize_-3)
    axs[0].grid(True)

    # Percentile subplot
    axs[1].scatter(sorted_swe_values, percentiles, color='blue', s=30, label='Empirical Data')
    axs[1].plot(x_fit, cdf_gumbel, color='black', linestyle='--', label='Gumbel Fit (MLE)')
    axs[1].set_xlabel('Maximum SWE (mm)', fontsize=fontsize_-3)
    axs[1].set_ylabel(r'Percentile (\%)', fontsize=fontsize_-3)
    axs[1].set_title(r'Percentile Plot with Gumbel Fit',
                     fontsize=fontsize_)
    axs[1].tick_params(axis='both', labelsize=fontsize_-4)
    axs[1].legend(fontsize=fontsize_-3)
    axs[1].grid(True)

    # Label subplots with (a) and (b) below the x-axis
    axs[0].text(0.5, -0.25, r"\textbf{(a)}", transform=axs[0].transAxes,
                fontsize=fontsize_, fontweight='bold', ha='center', va='top')

    axs[1].text(0.5, -0.25, r"\textbf{(b)}", transform=axs[1].transAxes,
                fontsize=fontsize_, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    output_filename = os.path.join(output_folder, 'swe_combined_plot.png')
    plt.savefig(output_filename, dpi=300)
    # plt.show()

def plot_quantile_mapping_demo():
    # Plot synthetic quantile mapping demo with time series and ECDF
    np.random.seed(42)
    n = 60
    time = np.arange(n)
    observed = np.cumsum(np.random.normal(0, 0.5, n)) + 20
    modelled_original = observed * 0.8 + np.random.normal(0, 0.5, n)

    ecdf_obs = ECDF(observed)
    ecdf_mod = ECDF(modelled_original)

    quantiles = np.linspace(0, 1, 100)
    percentiles_obs = np.percentile(observed, quantiles * 100)
    percentiles_mod = np.percentile(modelled_original, quantiles * 100)

    transfer_func = interp1d(percentiles_mod, percentiles_obs, bounds_error=False, fill_value="extrapolate")
    modelled_adjusted = transfer_func(modelled_original)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Time series
    axs[0].plot(time, observed, color='green', label='Observed')
    axs[0].plot(time, modelled_original, color='red', label='Modelled, original')
    axs[0].plot(time, modelled_adjusted, color='blue', label='Modelled, adjust')
    axs[0].set_title(r'Timeseries', fontsize=fontsize_-2, weight='bold')
    axs[0].set_xlabel('Time', fontsize=fontsize_-4)
    axs[0].set_ylabel('Variable (unit)', fontsize=fontsize_-4)
    axs[0].legend(fontsize=fontsize_-4)
    axs[0].grid(True)

    # ECDF plot
    axs[1].plot(percentiles_mod, quantiles, 'r-', label='Modelled ECDF')
    axs[1].plot(percentiles_obs, quantiles, 'g-', label='Observed ECDF')
    axs[1].set_title(r'Empirical Cumulative Distribution Functions', fontsize=fontsize_-2, weight='bold')
    axs[1].set_xlabel('Variable (unit)', fontsize=fontsize_-4)
    axs[1].set_ylabel(r'$F(x)$', fontsize=fontsize_-4)
    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=fontsize_-4)
    axs[1].grid(True)

    plt.tight_layout()
    output_filename = os.path.join(output_folder, 'quantile_mapping_demo.png')
    plt.savefig(output_filename, dpi=300)
    # plt.show()


#plot_quantile_mapping_demo()
#plot_combined_timeseries_and_percentile()

