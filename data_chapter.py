########################### Adding map to causal diagram ##################################
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


from statsmodels.tsa.seasonal import seasonal_decompose
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def ncfiles_to_series(nc_file_name, var_name, time_period = "Past"):
# Open the NetCDF file
    file_path = f'copernicus_raw/' + nc_file_name
    ds = xr.open_dataset(file_path)
    var = ds.to_dataframe()
    # Convert the 'time' column to datetime type
    var['time_bnds'] = pd.to_datetime(var['time_bnds'])
    # Group the DataFrame by the date and calculate the mean of 'ts' for each date
    var = var.groupby(var['time_bnds'].dt.date)[var_name].mean()
    date_past = pd.date_range(start='1950-01-01', end='2014-12-01', freq='MS')
    date_past = date_past.strftime('%Y-%m')
    date_future = pd.date_range(start='2015-01-01', end='2100-12-01', freq='MS')
    date_future = date_future.strftime('%Y-%m')
    if time_period == 'Past':
        var = var[:-1]
        var.index = date_past
    else:
        var = var[:-1]
        var.index = date_future

    return var

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

red_patch = mpatches.Patch(color='pink', alpha = 0.4, label='WPAC, CPAC & EPAC', hatch='\\\\\\', edgecolor='black')
green_patch = mpatches.Patch(color='lightblue',alpha = 0.4, label='Niño 3.4 region', hatch='..', edgecolor='black')
plum_patch = mpatches.Patch(color='plum',alpha = 0.4, label='EWW & THF', hatch='///', edgecolor='black')


# Create Plate Carrée projection
proj = ccrs.PlateCarree(central_longitude=180.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas
x, y = [-180, -130, -130, -180], [0, 0, -10, -10]
ax.fill(x, y, transform=central_proj, color='plum', alpha=0.4, zorder=0, hatch='///', edgecolor='black')
x, y = [-170, -120, -120, -170], [2.5, 2.5, -7.5, -7.5]
ax.fill(x, y, transform=central_proj, color='lightblue', alpha=0.4, zorder=0, hatch='..', edgecolor='black')
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, zorder=0, hatch='\\\\\\', edgecolor='black')
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4,zorder=0, hatch='\\\\\\', edgecolor='black')
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, hatch='\\\\\\', edgecolor='black')
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)

ax.set_extent([80, -120, -40, 40], crs=ccrs.PlateCarree(central_longitude=-140.0))
plt.legend(handles=[red_patch, green_patch, plum_patch])
ax.set_yticks([-40, -20, 0, 20, 40], crs=ccrs.PlateCarree())

plt.show()

epac_past = ncfiles_to_series('tas_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19500116-20141216_v20210113.nc', 'tas', time_period = "Past")
epac_future = ncfiles_to_series('tas_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113.nc', 'tas', time_period = "Future")
epac = epac_past.append(epac_future)

cpac_past = ncfiles_to_series('tas_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19500116-20141216_v20210113_2.nc ', 'tas', time_period = "Past")
cpac_future = ncfiles_to_series('tas_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113_2.nc', 'tas', time_period = "Future")
cpac = cpac_past.append(cpac_future)

wpac_past = ncfiles_to_series('psl_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19500116-20141216_v20210113.nc', 'psl', time_period = "Past")
wpac_future = ncfiles_to_series('psl_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113.nc', 'psl', time_period = "Future")
wpac = wpac_past.append(wpac_future)

nino34_past = ncfiles_to_series('tos_Omon_EC-Earth3-CC_historical_r1i1p1f1_gn_19500116-20141216_v20210113.nc', 'tos', time_period = "Past")
nino34_future = ncfiles_to_series('tos_Omon_EC-Earth3-CC_ssp585_r1i1p1f1_gn_20150116-21001216_v20210113.nc', 'tos', time_period = "Future")
nino34 = nino34_past.append(nino34_future)

eww_past = ncfiles_to_series('uas_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19500116-20141216_v20210113.nc', 'uas', time_period = "Past")
eww_future = ncfiles_to_series('uas_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113.nc', 'uas', time_period = "Future")
eww = eww_past.append(eww_future)

lhf_past = ncfiles_to_series('hfls_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_19500116-20141216_v20210113.nc', 'hfls', time_period = "Past")
lhf_future = ncfiles_to_series('hfls_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113.nc', 'hfls', time_period = "Future")
lhf = lhf_past.append(lhf_future)

shf_past = ncfiles_to_series('hfss_Amon_EC-Earth3-Veg-LR_historical_r1i1p1f1_gr_19500116-20141216_v20200217.nc', 'hfss', time_period = "Past")
shf_future = ncfiles_to_series('hfss_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_20150116-21001216_v20210113.nc', 'hfss', time_period = "Future")
shf = shf_past.append(shf_future)

#some data transformations

eww = eww * -1
thf = lhf + shf

cpac = deTrend_deSeasonalize(cpac, show_plot = False)
epac = deTrend_deSeasonalize(epac, show_plot = False)
wpac = deTrend_deSeasonalize(wpac, show_plot = False)
eww = deTrend_deSeasonalize(eww, show_plot = False)
thf = deTrend_deSeasonalize(thf, show_plot = False)

date = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS')
date = date.strftime('%Y-%m')

df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac, 'EWW': eww, 'THF': thf})
df = df[df['Date'] <= '2025-01']
var_names = ['WPAC', 'CPAC', 'EPAC', 'EWW', 'THF']

datatime = np.linspace(1950, 2025-1./12., len(df))
dataframe = pp.DataFrame(np.copy(df.iloc[:,1:6]), datatime = datatime, var_names= var_names)


from statsmodels.tsa.stattools import adfuller

adfuller(df['WPAC'], regression='n', autolag='AIC')
adfuller(df['CPAC'], regression='n', autolag='AIC')
adfuller(df['EPAC'], regression='n', autolag='AIC')
adfuller(df['EWW'], regression='n', autolag='AIC')
adfuller(df['THF'], regression='n', autolag='AIC')

df[var_names].describe()
df[var_names].median()

# Diagnostic plots

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





# repeat for future data 2025-2100

df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac, 'EWW': eww, 'THF': thf})
df = df[df['Date'] >= '2025-01']

datatime = np.linspace(2025, 2100-1./12., len(df))
dataframe = pp.DataFrame(np.copy(df.iloc[:,1:6]), datatime = datatime, var_names= var_names)

# Diagnostic plots

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
