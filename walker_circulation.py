

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# plt.style.use('ggplot')

import pkg_resources

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# to check tigramite version
print(pkg_resources.get_distribution("tigramite").version)

# Reading the data
cpac = pd.read_csv('walker_data_obs/iera5_t2m_-150--130E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=20, header=None)
epac = pd.read_csv('walker_data_obs/iera5_t2m_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=20, header=None)
wpac = pd.read_csv('walker_data_obs/iera5_slp_130-150E_-5-5N_n.dat', delimiter=r"\s+", skiprows=20, header=None)

cpac = pd.read_csv('walker_data/icmip6_tas_mon_modmean_ssp126_-150--130E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=58, header=None)
epac = pd.read_csv('walker_data/icmip6_tas_mon_modmean_ssp126_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=58, header=None)
wpac = pd.read_csv('walker_data/icmip6_psl_mon_modmean_ssp126_130-150E_-5-5N_n.dat', delimiter=r"\s+", skiprows=57, header=None)

cpac.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
epac.columns = cpac.columns
wpac.columns = cpac.columns


# stack observations into a single column
cpac = cpac.drop(cpac.columns[0], axis=1)
cpac = cpac.stack().reset_index(drop=True)

epac = epac.drop(epac.columns[0], axis=1)
epac = epac.stack().reset_index(drop=True)

wpac = wpac.drop(wpac.columns[0], axis=1)
wpac = wpac.stack().reset_index(drop=True)


# specify date and add to dataframe
date = pd.date_range(start='1950-01-01', end='2023-12-01', freq='MS')
date = date.strftime('%Y-%m')

df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})
df = df[df['Date'] <= '2023-03']
df = df[df['Date'] >= '1950-01']



# function that extracts seasonal component from a simple additive decomposition model, and subracts
# long term trend calculated with the Gaussian kernel smoother
def deTrend_deSeasonalize(data, show_plot = True):
    # step 1: seasonal component
    decomposition = seasonal_decompose(data, model='additive', period=12)
    data = data - decomposition.seasonal
    # step 2: Trend based on Gaussian kernel smoother
    smoothed = pp.smooth(np.copy(data), smooth_width=15 * 12, residuals=False)
    index = range(0, len(data))

    if show_plot == True:
        plt.plot(index, data)
        plt.plot(index,smoothed)
        plt.legend(['data', 'Smoothed trend'])
        plt.title("Deseasonalized data with smoothed trend")
        plt.show()

    else:
        result = data - smoothed
        return result



var_names = ['WPAC', 'CPAC', 'EPAC']
dataframe = pp.DataFrame(np.copy(df.iloc[:,1:4]), datatime = {0:np.arange(len(df))}, var_names= var_names)
tp.plot_timeseries(dataframe, color='black', show_meanline=True)
plt.show()

# transform data to bimonthly, to average out noisy monthly data
#data, _ = pp.time_bin_with_mask(np.copy(data),time_bin_length=2, mask=None)

deTrend_deSeasonalize(df['CPAC'], show_plot = True)
deTrend_deSeasonalize(df['EPAC'], show_plot = True)
plt.show()

deTrend_deSeasonalize(df['WPAC'], show_plot = True)
plt.show()


cpac = deTrend_deSeasonalize(df['CPAC'], show_plot = False)
epac = deTrend_deSeasonalize(df['EPAC'], show_plot = False)
wpac = deTrend_deSeasonalize(df['WPAC'], show_plot = False)
# double check
plt.plot(cpac)
plt.show()

plt.plot(epac)
plt.show()

plt.plot(wpac)
plt.show()




# we use the El niño 3.4 index to mask the data, so we can choose only the data during the Nov-Feb period with La Niña-Neutral conditions
# Oceanic Nino Index defined by 5 consecutive months of 3-month-running-mean of Nino3.4 SST above/below 0.5
start_year = 1950
length = 12*(2023 - start_year) +3

nino34 = np.loadtxt('climate_data/iersst_nino3.4a.txt', skiprows=44)
nino34= nino34[nino34[:,0] >= 1950]
#nino34= nino34[nino34[:,0] < 2023]
nino34smoothed = pp.smooth(nino34[:,1], smooth_width=3, kernel='heaviside',
           mask=None, residuals=False)

plt.plot(nino34[:,1])
plt.plot(nino34smoothed)
plt.show()

# Construct mask for only neutral and La Nina phases
nino_mask = np.zeros(length)
for t in range(length):
    if np.sum(nino34smoothed[max(0, t-4): min(length, t+5)] > 0.5) >= 5:
        nino_mask[t] = 1

# Time-bin mask since we will use bimonthly time series. If we change this later, remember to change cycle_length = 6
# nino_mask, _ = pp.time_bin_with_mask(nino_mask,
#   time_bin_length=2, mask=None)

# Construct mask to only select November to February
cycle_length = 12
mask = np.ones(df.shape, dtype='bool')
for i in [0, 1, 10, 11]:  #
    mask[i::cycle_length, :] = False

# Additionally mask to pick only neutral and La Nina phases
for t in range(mask.shape[0]):
    if nino_mask[t] >= 0.5:
        mask[t] = True



df = pd.DataFrame({'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})


# create tigramite dataframe to inspect
dataframe = pp.DataFrame(df.values, datatime = {0:np.arange(len(df))}, var_names= var_names)
# tigramite data inspection
# plot 1: timeseries
tp.plot_timeseries(dataframe, color='black', show_meanline=True)
plt.show()

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
results = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=None, alpha_level=0.01, fdr_method = 'fdr_bh')

# plot 5: causal graph
tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI')
plt.show()

########################### Adding map to causal diagram ##################################

import cartopy
import cartopy.crs as ccrs



# Create Plate Carrée projection
proj = ccrs.Orthographic(central_longitude=-155.0,central_latitude=15.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4,zorder=0)
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)


# causal network plot

node_pos = {
    'y': np.array([0., 0., 0.]),
    'x': np.array([-5000000., 1000000., 6000000.])
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

########################### Transform data and do the same analysis ##################################

from statsmodels.tsa.seasonal import seasonal_decompose

# Subset only data from 1970-2020
df = df[df['Date'] >= '1970-01']

decompositionWPAC = seasonal_decompose(df['WPAC'], model='additive', period=12)
decompositionCPAC = seasonal_decompose(df['CPAC'], model='additive', period=12)
decompositionEPAC = seasonal_decompose(df['EPAC'], model='additive', period=12)

decompositionWPAC.plot()
plt.show()

# specify new date
date = pd.date_range(start='1970-01-01', end='2020-12-01', freq='MS')
date = date.strftime('%Y-%m')
# deseasonalize
WPAC_deseason = df['WPAC'] - decompositionWPAC.seasonal
CPAC_deseason = df['CPAC'] - decompositionCPAC.seasonal
EPAC_deseason = df['EPAC'] - decompositionEPAC.seasonal


# create new dataframe with deseasonalized data
df_deseason = pd.DataFrame({'Date': date, 'WPAC': WPAC_deseason,'CPAC': CPAC_deseason, 'EPAC':EPAC_deseason})

dataframe = pp.DataFrame(df_deseason.iloc[:, 1:4].values, datatime = {0:np.arange(len(df_deseason))}, var_names= var_names)

# rerun code from before(tigramite data inspection)

# aggregate over the year
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year

df_year = df.groupby('year').mean()
df_year = df_year[df_year.index >= 1970]
dataframe = pp.DataFrame(df_year.iloc[:, 0:3].values, datatime = {0:np.arange(len(df_year))}, var_names= var_names)

# rerun code from before(tigramite data inspection)