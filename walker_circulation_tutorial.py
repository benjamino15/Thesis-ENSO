import numpy as np
import matplotlib.pyplot as plt
import sys

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


import tigramite
import tigramite.data_processing as pp
import tigramite.plotting as tp

from tigramite.models import LinearMediation, Models
from tigramite.causal_effects import CausalEffects
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# Pick time range
start_year = 1950
length = 12*(2022 - start_year) + 3

# Load the three climate time series
# (these were generated with the KNMI climate explorer, see the headers of the files for details)
data = np.vstack((
    np.loadtxt('climate_data/iera5_w700_130-150E_20-0N_n.dat', skiprows=20)[:, 1:].reshape(-1)[:length],
    np.loadtxt('climate_data/iera5_u10m_160-180E_5--5N_n.dat', skiprows=20)[:, 1:].reshape(-1)[:length],
    np.loadtxt('climate_data/iera5_w700_-160--120E_5--5N_n.dat', skiprows=20)[:, 1:].reshape(-1)[:length],
                )).T
T, N = data.shape

# Simple way to define time axis
datatime = np.linspace(start_year, 2023-1./12., T)

# Variable names used throughout
var_names = ['WPAC', 'WCPAC', 'CPAC']

# Time-bin data and datatime. Transform monthly data into bi-monthly, to average-out noisy monthly data
data, _ = pp.time_bin_with_mask(data,time_bin_length=2, mask=None)
datatime, _ = pp.time_bin_with_mask(datatime,time_bin_length=2, mask=None)

# Remove trend and seasonal component, in order to make the data stationary

# Function to remove seasonal mean and divide by seadonal standard deviation
def anomalize(dataseries, divide_by_std=True, reference_bounds = None, cycle_length=12, return_cycle=False):
    if reference_bounds is None:
        reference_bounds = (0, len(dataseries))

    anomaly = np.copy(dataseries)
    for t in range(cycle_length):
        if return_cycle:
            anomaly[t::cycle_length] = dataseries[t+reference_bounds[0]:reference_bounds[1]:cycle_length].mean(axis=0)
        else:
            anomaly[t::cycle_length] -= dataseries[t+reference_bounds[0]:reference_bounds[1]:cycle_length].mean(axis=0)
            if divide_by_std:
                anomaly[t::cycle_length] /= dataseries[t+reference_bounds[0]:reference_bounds[1]:cycle_length].std(axis=0)
    return anomaly

cycle_length = 6
smooth_width_effects = 15*cycle_length

if smooth_width_effects is not None:
    smoothdata_here = pp.smooth(np.copy(data), smooth_width=smooth_width_effects, kernel='gaussian',
                    residuals=False)
    data_here = pp.smooth(np.copy(data), smooth_width=smooth_width_effects, kernel='gaussian',
                    residuals=True)
else:
    print("Not smoothed.")
    data_here = np.copy(data)

# Remove seasonal mean and divide by seasonal standard deviation
seasonal_cycle = anomalize(np.copy(data_here), cycle_length=cycle_length, return_cycle=True)
smoothdata_here += seasonal_cycle



# we use the El niño 3.4 index to mask the data, so we can choose only the data during the Nov-Feb period with La Niña-Neutral conditions
# Oceanic Nino Index defined by 5 consecutive months of 3-month-running-mean of Nino3.4 SST above/below 0.5
nino34 = np.loadtxt('climate_data/iersst_nino3.4a.txt', skiprows=44)[(start_year-1854)*12:(start_year-1854)*12+length,1]
nino34smoothed = pp.smooth(nino34, smooth_width=3, kernel='heaviside',
           mask=None, residuals=False)

# Construct mask for only neutral and La Nina phases
nino_mask = np.zeros(length)
for t in range(length):
    if np.sum(nino34smoothed[max(0, t-4): min(length, t+5)] > 0.5) >= 5:
        nino_mask[t] = 1

# Time-bin mask since we will use bimonthly time series
nino_mask, _ = pp.time_bin_with_mask(nino_mask,
    time_bin_length=2, mask=None)

# Construct mask to only select November to February
# (cycle_length is the length of the year in the bimonthly time series)
cycle_length = 6
mask = np.ones(data.shape, dtype='bool')
for i in [0, 5]:  #
    mask[i::cycle_length, :] = False

# Additionally mask to pick only neutral and La Nina phases
for t in range(mask.shape[0]):
    if nino_mask[t] >= 0.5:
        mask[t] = True


# Dataframe for raw data
raw_dataframe = pp.DataFrame(np.copy(data), mask=mask, var_names=var_names, datatime=datatime)

# Dataframe for smoothed data
smoothdataframe_here = pp.DataFrame(smoothdata_here, var_names=var_names,  datatime=datatime)


fig, axes = tp.plot_timeseries(
        raw_dataframe,
        figsize=(6, 4),
        grey_masked_samples='data',
        color='black',
        show_meanline=True,
        adjust_plot=False,
        )

tp.plot_timeseries(
        smoothdataframe_here,
        fig_axes = (fig, axes),
        grey_masked_samples='data',
        show_meanline=False,
        color='red',
        alpha=0.4,
        adjust_plot=True,
        tick_label_size=7,
        label_fontsize=8,
        time_label='year',
        var_units=['Pa/s', 'm/s', 'Pa/s',])
plt.show()

if smooth_width_effects is not None:
    data_here = pp.smooth(data, smooth_width=smooth_width_effects, kernel='gaussian',
                    residuals=True)
else:
    data_here = np.copy(data)

data_here = anomalize(data_here, cycle_length=cycle_length)

# Initialize Tigramite dataframe
dataframe = pp.DataFrame(data_here, mask=mask, var_names=var_names, missing_flag=999.)


parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1)
correlations = pcmci.get_lagged_dependencies(tau_max=20, val_only=True)['val_matrix']

# plot 2: scatterplots
matrix_lags = None
tp.plot_scatterplots(dataframe=dataframe)
plt.show()

# plot 3: marginal and joint densities
tp.plot_densityplots(dataframe=dataframe, add_densityplot_args={'matrix_lags':matrix_lags})
plt.show()

# plot 4: lag functions
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names,
                                    'x_base':5, 'y_base':.5}); plt.show()

# run PCMCI
pcmci.verbosity = 1
results = pcmci.run_pcmci(tau_max=2, tau_min=1, pc_alpha=None, alpha_level=0.01)

# plot 5: causal graph

import cartopy
import cartopy.crs as ccrs



# Create Plate Carrée projection
proj = ccrs.Orthographic(central_longitude=-170.0, central_latitude=15.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas
x, y = [130, 140, 150, 150, 140, 130], [0, 0, 0, 20, 20, 20]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
x, y = [160, 170, 180, 180, 170, 160], [-5, -5, -5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4,zorder=0)
x, y = [-160, -150, -140, -130, -120, -120, -130, -140, -150, -160], [-5, -5, -5, -5, -5, 5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)


# causal network plot

node_pos = {
    'y': np.array([0., -1600000., -1400000.]),
    'x': np.array([-4700000., -2000000., 3300000.])
}

ax = tp.plot_graph(
    fig_ax=(fig, ax),
    graph = results['graph'],
    node_pos=node_pos,
    figsize=(10, 5),
    val_matrix=results['val_matrix'],
    #cmap_edges='RdBu_r',
    #edge_ticks=.5,
    #show_colorbar=False,
    var_names=var_names,
    arrow_linewidth= 4,
    curved_radius=.35,
    node_size=1000000.,
    #node_label_size=8,
    node_aspect= 1.,
    #link_label_fontsize=0,
    #label_fontsize=8,
    node_colorbar_label='auto-MCI',
    link_colorbar_label='cross-MCI')

plt.show()