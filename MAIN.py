import print_to_csv
import plot_map
import plot_scatter

# All data storrage to csv files of SWE values from SeNorge, results from calculations, points and elevations
# along with all relevant plots can be produced from running all lines in this file

## Points and Elevation ##

#print_to_csv.print_to_csv_points()
#print_to_csv.print_to_csv_elevation()

## SWE ##

#print_to_csv.print_to_csv_swe("tot")
#print_to_csv.print_to_csv_swe("old")
#print_to_csv.print_to_csv_swe("new")
#print_to_csv.print_to_csv_swe("future_rcp45")
#print_to_csv.print_to_csv_swe("future_rcp85")

## Reliaiblity indices ##

#print_to_csv.print_to_csv_beta("tot")
#print_to_csv.print_to_csv_beta("old")
#print_to_csv.print_to_csv_beta("new")
#print_to_csv.print_to_csv_beta("future_rcp45")
#print_to_csv.print_to_csv_beta("future_rcp85")


## Difference, optimal, char_ec and CoV ##

#print_to_csv.print_to_csv_char()
#print_to_csv.print_to_csv_cov("tot")
#print_to_csv.print_to_csv_cov("future_rcp45")
#print_to_csv.print_to_csv_cov("future_rcp85")
#print_to_csv.print_to_csv_char_opt("tot")
#print_to_csv.print_to_csv_beta_opt("tot")
#print_to_csv.print_to_csv_char_opt("future_rcp45")
#print_to_csv.print_to_csv_char_opt("future_rcp85")
#print_to_csv.print_to_csv_diff("new", "beta", "old", "beta")
#print_to_csv.print_to_csv_diff("tot", "opt_char", "ec", "char")
#print_to_csv.print_to_csv_diff("future_rcp45", "opt_char", "ec", "char")
#print_to_csv.print_to_csv_diff("future_rcp85", "opt_char", "ec", "char")
#print_to_csv.print_to_csv_T50_char()
#print_to_csv.print_to_csv_T50_beta()


## Plot ##
 
#plot_map.map("tot", "beta")
#plot_map.map("new", "beta")
#plot_map.map("old", "beta", show_colorbar=False)
#plot_map.map("future_rcp45", "beta")
#plot_map.map("future_rcp85", "beta")
#plot_map.map("new_old", "diff_beta")
#plot_map.map("tot", "opt_char")
#plot_map.map("tot", "opt_beta")
#plot_map.map("tot", "T50_char")
#plot_map.map("tot", "T50_beta")
#plot_map.map("future_rcp45", "opt_char")
#plot_map.map("future_rcp85", "opt_char")
#plot_map.map("tot", "cov")
#plot_map.map("future_rcp45", "cov")
#plot_map.map("future_rcp85", "cov")

plot_scatter.scatter("tot")
#plot_scatter.scatter("new")
#plot_scatter.scatter("old")
#plot_scatter.scatter("future_rcp45")
#plot_scatter.scatter("future_rcp85")
#plot_scatter.scatter_char_violin("tot")
#plot_scatter.scatter_char_violin("future_rcp45")
#plot_scatter.scatter_char_violin("future_rcp85")

#plot_scatter.scatter_char_box("tot")
#plot_scatter.scatter_char_box("future_rcp45")
#plot_scatter.scatter_char_box("future_rcp85")



