import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from C_Input_AHG import input_data
from FORM import prop
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # Matches LaTeX default
    "axes.formatter.use_mathtext": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm

# This file produces scatter and violin plots, as well as box plots. The box plots are not used 
# in the thesis, but they are a good alternative to the violin plots. In the scatter plots it is 
# differentiated between reliailbity indices produced from prescribed characteristic values and 
# the ones calculated based on SWE values.

def scatter(time, T50=None):
    
    fontsize_ = 26

    # File paths
    if T50:
        beta_file = f"stored_data/T50_beta_{time}.csv"

    else:
        beta_file = f"stored_data/beta_{time}.csv"
    
    swe_file = f"stored_data/swe_{time}.csv"

    output_folder = f"/Users/hakon/SnowAnalysis_HU/Figures/main_output/"
    
    if T50:
        output_combined_file = output_folder + f"scatter_T50_{time}.png"
    else:
        output_combined_file = output_folder + f"scatter_{time}.png"

    # Load data
    beta_df = pd.read_csv(beta_file)
    beta_df.rename(columns={"var": "variable"}, inplace=True)
    swe_df = pd.read_csv(swe_file)

    # Set Municipality as index
    beta_df.set_index("municipality", inplace=True)
    swe_df.set_index("municipality", inplace=True)

    # Convert SWE column from string to list
    def parse_swe(value):
        try:
            return ast.literal_eval(value) if isinstance(value, str) else value
        except (SyntaxError, ValueError):
            return []

    swe_df["swe"] = swe_df["var"].apply(parse_swe)

    def gumbel_params(swe_list):
        if not swe_list or np.sum(swe_list) < 10:
            return pd.Series([np.nan, np.nan])
        loc, scale = stats.gumbel_r.fit(swe_list)
        gamma = 0.57722
        mean_gumbel = loc + gamma * scale
        std_gumbel = (np.pi / np.sqrt(6)) * scale
        cov = std_gumbel / mean_gumbel
        mean_kN_per_m2 = mean_gumbel * 9.8 * 2 / 1000  # Convert from mm SWE to kN/m²
        return pd.Series([mean_kN_per_m2, cov])

    swe_df[["SWE_Gumbel_Mean", "SWE_Gumbel_CoV"]] = swe_df["swe"].apply(gumbel_params)

    # Merge datasets
    swe_df = swe_df.drop(columns=["var"])  # Add this line before the join
    merged_df = swe_df.join(beta_df, how="inner").dropna(subset=["SWE_Gumbel_Mean", "SWE_Gumbel_CoV", "variable"])
    merged_df = merged_df[merged_df["variable"] != 12]

    if not T50:
        # Remove Mean Effect
        X_mean = merged_df[["SWE_Gumbel_Mean"]]
        y_reliability = merged_df["variable"]
        # The mean doesn't seem in a linear relationship with reliaiblity, but was chosen as a simplification
        model_mean = LinearRegression()
        model_mean.fit(X_mean, y_reliability)
        merged_df["Residual_Reliability"] = y_reliability - model_mean.predict(X_mean)

    # Create Subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)  # sharey=True if axes should align

    # Plot 1: Reliability vs. Mean
    axs[0].scatter(
        merged_df["SWE_Gumbel_Mean"],
        merged_df["variable"],
        s=20,
        alpha=1
    )

    if input_data[time]['scenario'] == None:
        string = ""
    elif input_data[time]['scenario'] == "rcp45":
        string = "RCP 4.5"
    else:
        string = "RCP 8.5"


    axs[0].axhline(y=3.8, color='red', linestyle='dashed', linewidth=1.5, label="Reliability target = 3.8")
    axs[0].set_xlabel("Mean [$kN/m^2$]", fontsize=fontsize_-4)
    axs[0].set_ylabel("Reliability Index", fontsize=fontsize_-4)
    axs[0].set_title(f"Reliability Index vs. Mean\n {input_data[time]['title']}", fontsize=fontsize_)
    axs[0].tick_params(labelsize=fontsize_-6)
    axs[0].grid(True)
    axs[0].legend(fontsize=fontsize_-4)

    # Plot 2: Residual vs. CoV 

    if T50:
        y_string = "variable"
    else:
        y_string = "Residual_Reliability"

    axs[1].scatter(
        merged_df["SWE_Gumbel_CoV"],
        merged_df[y_string],
        s=20,
        alpha=1
    )
    axs[1].set_xlabel("CoV", fontsize=fontsize_-4)
    if not T50:
        axs[1].set_ylabel("Residual Reliability Index", fontsize=fontsize_-4)
    if T50:
        axs[1].set_title(f"Reliability Index vs. CoV\n {input_data[time]['title']}", fontsize=fontsize_)
    else:
        axs[1].set_title(f"Residual Reliability vs. CoV\n {input_data[time]['title']}", fontsize=fontsize_)
    axs[1].tick_params(labelsize=fontsize_-6)
    axs[1].grid(True)


    # Label subplots with (a) and (b) below the x-axis
    axs[0].text(0.5, -0.25, r"\textbf{(a)}", transform=axs[0].transAxes,
                fontsize=fontsize_, fontweight='bold', ha='center', va='top')

    axs[1].text(0.5, -0.25, r"\textbf{(b)}", transform=axs[1].transAxes,
                fontsize=fontsize_, fontweight='bold', ha='center', va='top')


    # Tight layout to avoid overlap
    plt.tight_layout()

    # Save combined plot
    #plt.show()
    plt.savefig(output_combined_file, dpi=300)
    plt.close()
    print(f"Combined scatter plot saved at: {output_combined_file}")





def scatter_char_box(time):
    fontsize_ = 25
    # Load data
    ec_path = f"/Users/hakon/SnowAnalysis_HU/stored_data/char_ec.csv"
    opt_path = f"/Users/hakon/SnowAnalysis_HU/stored_data/opt_char_{time}.csv"
    output_path = f"/Users/hakon/SnowAnalysis_HU/Figures/main_output/scatter_char_box_{time}.png"

    ec_df = pd.read_csv(ec_path)
    opt_df = pd.read_csv(opt_path)

    # Set Municipality as index
    ec_df.set_index("municipality", inplace=True)
    opt_df.set_index("municipality", inplace=True)

    # Merge with suffixes
    merged_df = ec_df.join(opt_df, how="inner", lsuffix="_EC", rsuffix="_opt")
    merged_df.rename(columns={"variable_EC": "Char_EC", "variable_opt": "Char_opt"}, inplace=True)

    # Drop missing values
    merged_df.dropna(subset=["Char_EC", "Char_opt"], inplace=True)

    # Convert Char_EC to categorical (grouping for boxplot)
    merged_df["Char_EC"] = merged_df["Char_EC"].round(2)  # optional: round to avoid tiny floating differences

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Char_EC", y="Char_opt", data=merged_df, palette="Set3")
    #sns.boxplot(x="Char_EC", y="Char_opt", data=merged_df, hue="Char_EC", palette="Set3", inner="box", legend=False)


    # Add 1:1 line
    unique_ec = sorted(merged_df["Char_EC"].unique())
    tick_positions = range(len(unique_ec))
    plt.plot(tick_positions, unique_ec, linestyle="--", color="gray", label="1:1 Line")
    plt.legend()


    plt.xlabel("Prescribed Characteristic Value", fontsize=fontsize_)
    plt.ylabel("Optimal Characteristic Value", fontsize=fontsize_)
    plt.title(f"Comparison of Optimal and Prescribed Characteristic Values {str(input_data[time]["period"])}", fontsize=fontsize_)
    plt.xticks(fontsize=fontsize_-5)
    plt.yticks(fontsize=fontsize_-5)

    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    #plt.show()

    print(f"Plot saved to: {output_path}")




def scatter_char_violin(time):
    fontsize_ = 30
    # File paths
    ec_path = f"/Users/hakon/SnowAnalysis_HU/stored_data/char_ec.csv"
    opt_path = f"/Users/hakon/SnowAnalysis_HU/stored_data/opt_char_{time}.csv"
    output_path = f"/Users/hakon/SnowAnalysis_HU/Figures/main_output/scatter_char_violin_{time}.png"

    # Load data
    ec_df = pd.read_csv(ec_path).rename(columns={"var": "variable"})
    opt_df = pd.read_csv(opt_path).rename(columns={"var": "variable"})

    ec_df.set_index("municipality", inplace=True)
    opt_df.set_index("municipality", inplace=True)

    # Merge and prepare
    merged_df = ec_df.join(opt_df, how="inner", lsuffix="_EC", rsuffix="_opt")
    merged_df.rename(columns={"variable_EC": "Char_EC", "variable_opt": "Char_opt"}, inplace=True)
    merged_df.dropna(subset=["Char_EC", "Char_opt"], inplace=True)
    merged_df["Char_EC"] = merged_df["Char_EC"].round(2)

    # Prepare data for beanplot
    categories = sorted(merged_df["Char_EC"].unique())

    # Prepare data for beanplot (skip groups with < 2 values)
    grouped = merged_df.groupby("Char_EC")["Char_opt"].apply(list)
    filtered = grouped[grouped.apply(lambda x: len(x) >= 2)]
    data_for_plot = filtered.tolist()
    category_labels = [str(cat) for cat in filtered.index]


    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(0, 20)
    ax.grid(axis='y')

    plot_opts = {
        "cutoff": True,
        "cutoff_type": "abs",
        "cutoff_val": 0,
        "violin_fc": (0.7, 0.7, 0.7),
        "label_rotation": 90,
        "bean_show_mean": True,
        "bean_show_median": False,
        "jitter_marker": '.',
        "jitter_marker_size": 1.0,
        "bean_legend_text": "Municipality"
    }

    sm.graphics.beanplot(data_for_plot, ax=ax, labels=category_labels,
                         jitter=True, plot_opts=plot_opts)
    
    # Add 1:1 red dashed line manually
    x0, y0 = 1, 1.5     # First violin's position and value
    xn, yn = 12,7.5  # Last violin's position and value

    ax.plot([x0, xn], [y0, yn], linestyle="--", color="red", label="1:1 Line")
    ax.legend(fontsize=fontsize_-11)


    ax.set_xlabel("Prescribed Characteristic Value", fontsize=fontsize_-4)
    ax.set_ylabel("Optimal Characteristic Value", fontsize=fontsize_-4)
    ax.set_title(f"Comparison of Optimal and Prescribed Characteristic Values\n {input_data[time]['title']}", fontsize=fontsize_)
    ax.tick_params(axis='x', labelsize=fontsize_ - 6)
    ax.tick_params(axis='y', labelsize=fontsize_ - 6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

    print(f"Plot saved to: {output_path}")




    


#Test
#scatter_char_box("tot")
#scatter_char_box("new")
#scatter_char_box("old")
#scatter_char_violin("tot")
#scatter("tot")
#scatter("tot", T50=True)
#scatter_char_violin("future_rcp85")