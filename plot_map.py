import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import csv
import os
from pyproj import Transformer
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches  # Import for custom legend patches
import ast
from C_Input_AHG import input_data
from matplotlib.lines import Line2D


# Latex font

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # Matches LaTeX default
    "axes.formatter.use_mathtext": True
})

# This file plots a map with values from a csv files for each municipality
# Colorbar can be removed. Can be useful is two plots are to be put side by side displaying the same variable

def map(time, variable, show_colorbar=True):
    # Font size control for both title and legend 
    fontsize_ = 21

    # To increase automation, input data are added to C_Input_AHG that gives relevant parameters
    # Depening on what variable is plotted, and what period is used as input

    limits = input_data[variable]["limits"]
    label_ = input_data[variable]["label"]
    title_ = f"{input_data[variable]['title']} {input_data[time]['title']}"

    # File paths
    geojson_path = "C:/Users/hakon/SnowAnalysis_HU/DataSources/Basisdata_0000_Norge_25833_Kommuner_GeoJSON/Basisdata_0000_Norge_25833_Kommuner_GeoJSON.geojson"
    output_map_path_1 = f"C:/Users/hakon/SnowAnalysis_HU/Figures/main_output/{variable}_{time}.pdf"
    output_map_path_2 = f"C:/Users/hakon/SnowAnalysis_HU/Figures/main_output/{variable}_{time}.png"
    var_csv_path = f"C:/Users/hakon/SnowAnalysis_HU/stored_data/{variable}_{time}.csv"
    points_csv_path = f"C:/Users/hakon/SnowAnalysis_HU/stored_data/points.csv"

    # Load data
    var_df = pd.read_csv(var_csv_path)
    points_df = pd.read_csv(points_csv_path)

    # Parse the string representation of tuples in 'points' column
    def extract_lat_lon(point_str):
        try:
            coords = ast.literal_eval(point_str)
            if coords and isinstance(coords[0], tuple):
                lat, lon = coords[0]
                return pd.Series([lat, lon])
            else:
                return pd.Series([None, None])
        except Exception:
            return pd.Series([None, None])

    points_df[["latitude", "longitude"]] = points_df["var"].apply(extract_lat_lon)

    # Merge the variable and coordinate data
    df = pd.merge(var_df, points_df[["municipality", "latitude", "longitude"]], on="municipality", how="inner")
    df = df[["municipality", "latitude", "longitude", "var"]]

    # Convert coordinates to UTM Zone 33 (EPSG:32633)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    df["Easting"], df["Northing"] = zip(*df.apply(lambda row: transformer.transform(row["longitude"], row["latitude"]), axis=1))

    # Load the GeoJSON file (only Kommune layer)
    gdf_kommune = gpd.read_file(geojson_path, layer="Kommune")

    # Ensure correct CRS
    if gdf_kommune.crs is None:
        gdf_kommune.set_crs(epsg=25833, inplace=True)
    elif gdf_kommune.crs.to_epsg() != 25833:
        gdf_kommune.to_crs(epsg=25833, inplace=True)

    # Find municipalities with the lowest and highest var
    min_beta_row = df.loc[df["var"].idxmin()]
    max_beta_row = df.loc[df["var"].idxmax()]

    # Define key municipalities to highlight
    highlight_municipalities = ["Oslo", "Bergen", "Trondheim"]
    highlight_rows = df[df["municipality"].isin(highlight_municipalities)]

    # Normalize beta values for colormap
    # Allows for automatic limits for variable, or a predefined. The latter is preferred for better control and to facilitate comparison across maps
    lim = 1
    if lim == 0:
        norm = Normalize(vmin=df["var"].min(), vmax=df["var"].max())
    else:
        norm = Normalize(vmin=limits[0], vmax=limits[1])

    colormap = plt.cm.RdYlGn
    df["Color"] = df["var"].apply(lambda beta: colormap(norm(beta)))

    # Geometry mapping
    df["Geometry"] = df.apply(lambda row: gpd.points_from_xy([row["Easting"]], [row["Northing"]])[0], axis=1)
    df["Municipality_GDF"] = df.apply(lambda row: gdf_kommune[gdf_kommune.intersects(row["Geometry"])], axis=1)

    # The possibility to check for missing municipalities, i.e. municipalities not included
    run_missing = 0

    if run_missing == 1:
        municipalities_with_points = set()
        for matched in df["Municipality_GDF"]:
            municipalities_with_points.update(matched["kommunenavn"].tolist())

        all_municipalities = set(gdf_kommune["kommunenavn"])
        municipalities_without_points = all_municipalities - municipalities_with_points

        print("Municipalities without any data points:")
        for name in sorted(municipalities_without_points):
            print("-", name)

    # Plotting 
    # Consistent map area, usefull to enable maps to be arranged side by side in latex without size mismatch
    # Subplot was considered, but this would decrease automation and increase complexity
    fig = plt.figure(figsize=(6.3, 8))
    ax = fig.add_axes([0.0, 0.02, 0.89, 0.9])  # [left, bottom, width, height]
        
    ax.set_title(title_, fontsize=fontsize_)

    gdf_kommune.plot(ax=ax, color='white', edgecolor=None, linewidth=0.0)

    for _, row in df.iterrows():
        if not row["Municipality_GDF"].empty:
            row["Municipality_GDF"].plot(ax=ax, color=row["Color"], edgecolor=None, linewidth=0.0)

    # Optional colorbar
    if show_colorbar:
        cax = fig.add_axes([0.89, 0.025, 0.02, 0.89])  # fixed position for colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label_, fontsize=fontsize_-3, labelpad=20, rotation=90)
        cbar.ax.yaxis.set_label_position('left')  # Moves label to the left of the colorbar
        cbar.ax.tick_params(labelsize=fontsize_-4)

    # Custom legend
    legend_entries = [
        (min_beta_row["municipality"], min_beta_row["var"]),
        (max_beta_row["municipality"], max_beta_row["var"])
    ] + list(zip(highlight_rows["municipality"], highlight_rows["var"]))

    legend_handles = [
        Line2D([0], [0],
            marker='o',
            color='w',
            label=f"{mun} ({beta:.1f})",
            markerfacecolor=colormap(norm(beta)),
            markersize=10)
        for mun, beta in legend_entries
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=fontsize_-4,
        frameon=True,
        handletextpad=0.001  # smaller value = smaller gap
    )

    # Remove grid and axis details 
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Save
    os.makedirs(os.path.dirname(output_map_path_1), exist_ok=True)
    #plt.savefig(output_map_path_1, dpi=300)
    plt.savefig(output_map_path_2, dpi=300)
    #plt.show()
    plt.close()

#map("tot", "beta")
#map("old", "beta", show_colorbar=False)
#map("tot", "opt_char")
#map("future_rcp45", "beta")

#map("tot", "char_actual")
#map("future_rcp85", "opt_char")