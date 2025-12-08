# Install package from github
#devtools::install_github("https://github.com/NateOConnellPhD/OpenClustered")

# load package
library(OpenClustered)

# List available datasets
data(package = "OpenClustered")

# View Meta Data files
# exclude the 6th column 'origin' for cleaner output
head(OpenClustered::meta_data)[,-7]
# View meta data characteristics of all datasets in `data_list`
plot_meta_data(allplots=T)
# Summarize Meta Data (using r package "table1")
tab_meta_data(~n_obs +  n_features + n_clusters + imbalance + missing_percent)
