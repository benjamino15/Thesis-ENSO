import xarray as xr
import cftime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


#var['time_bnds'] = var['time_bnds'].apply(convert_to_datetime)
nc_file_name = 'MIROC6/tas_Amon_MIROC6_ssp585_20150116-21001216.nc'
var_name = 'tas'
def ncfiles_to_series(nc_file_name, var_name, time_period = "Past"):
# Open the NetCDF file
    file_path = f'copernicus_raw/' + nc_file_name
    ds = xr.open_dataset(file_path, engine = 'netcdf4')
    var = ds.to_dataframe()
    try:
            var['time_bnds'] = pd.to_datetime(var['time_bnds'])
        except TypeError:
            # Handle the TypeError here
            # Execute alternative code or raise an exception
            ds.to_dataframe().to_csv('copernicus/' + nc_file_name)
            var = pd.read_csv('copernicus/' + nc_file_name)
            var['time_bnds'] = pd.to_datetime(var['time_bnds'])

    # Group the DataFrame by the date and calculate the mean of 'ts' for each date
    try:
            var = var.groupby(var['time_bnds'].dt.date)[var_name].mean()
        except KeyError:
            var = var.groupby(var['time_bnds'].dt.date)['tas'].mean()

    date_past = pd.date_range(start='1950-01-01', end='2014-12-01', freq='MS')
    date_past = date_past.strftime('%Y-%m')
    date_future = pd.date_range(start='2015-01-01', end='2100-12-01', freq='MS')
    date_future = date_future.strftime('%Y-%m')
    if len(var) > 1500:
        var.index = pd.to_datetime(var.index)
        var = var.resample('M').mean()
    else:

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

def comparison_metrics(reference_val_matrix, reference_p_matrix, val_matrix, p_matrix, threshold):
    mask_array = np.where(reference_p_matrix < threshold, 1, 0)
    mask_array = mask_array.flatten().tolist()
    reference_val_matrix = reference_val_matrix.flatten().tolist()
    val_matrix = val_matrix.flatten().tolist()
    abs_difference = []
    squared_difference = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1:
            abs_difference.append(abs(reference_val_matrix[i] - val_matrix[i]))
            squared_difference.append((reference_val_matrix[i] - val_matrix[i])**2)

    mask_array2 = np.where(p_matrix < threshold, 1, 0)
    mask_array2 = mask_array2.flatten().tolist()
    true_positives = []
    false_positives = []
    false_negatives = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1 and mask_array2[i] == 1:
            true_positives.append(1)
        elif mask_array[i] == 1 and mask_array2[i] == 0:
            false_negatives.append(1)
        elif mask_array[i] == 0 and mask_array2[i] == 1:
            false_positives.append(1)

    precision = np.sum(true_positives)/(np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives)/(np.sum(true_positives) + np.sum(false_negatives))
    f1_score = (2*precision * recall)/(precision + recall)
    return np.mean(abs_difference), np.sum(squared_difference)**0.5, np.sum(false_negatives), np.sum(false_positives), f1_score

def link_difference(reference_val_matrix, reference_p_matrix, val_matrix, p_matrix, threshold):
    mask_array = np.where(reference_p_matrix < threshold, 1, 0)
    mask_array2 = np.where(p_matrix < threshold, 1, 0)
    mask_array = mask_array.flatten().tolist()
    mask_array2 = mask_array2.flatten().tolist()
    auto_coefficients = []
    for s in range(0,5):
        auto_coefficients.append(list(range(s + 23*s,s+4 + 23*s)))

    auto_coefficients_flatten = [item for sublist in auto_coefficients for item in sublist]
    auto_coefficients = auto_coefficients_flatten

    for i in range(len(mask_array)):
        if i in auto_coefficients:
          mask_array[i] = 0
          mask_array2[i] = 0
    reference_val_matrix = reference_val_matrix.flatten().tolist()
    val_matrix = val_matrix.flatten().tolist()
    difference = []
    difference2 = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1:
            difference.append(abs(reference_val_matrix[i]))
        if mask_array2[i] == 1:
            difference2.append(abs(val_matrix[i]))
    return (np.mean(difference) - np.mean(difference2))
def run_pcmci(wpac, cpac, epac, nino34, eww, thf, start = '1950-01', end = '2025-01'):

    # specify date and add to dataframe
    date = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS')
    date = date.strftime('%Y-%m')

    df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac, 'EWW': eww, 'THF': thf})
    df = df[df['Date'] <= end]
    df = df[df['Date'] >= start]

    var_names = ['WPAC', 'CPAC', 'EPAC', 'EWW', 'THF']
    cpac = deTrend_deSeasonalize(df['CPAC'], show_plot=False)
    epac = deTrend_deSeasonalize(df['EPAC'], show_plot=False)
    wpac = deTrend_deSeasonalize(df['WPAC'], show_plot=False)
    eww = deTrend_deSeasonalize(df['EWW'], show_plot=False)
    thf = deTrend_deSeasonalize(df['THF'], show_plot=False)

    df_nino34 = pd.DataFrame({'Date': date, 'nino34': nino34})
    df_nino34 = df_nino34[df_nino34['Date'] <= end]
    df_nino34 = df_nino34[df_nino34['Date'] >= start]
    nino34 = df_nino34['nino34']
    #detrend nino34 index
    nino34 = pp.smooth(np.copy(nino34), smooth_width=30 * 12, residuals=True)

    #start analysis
    df = pd.DataFrame({'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac, 'EWW': eww, 'THF': thf})

    # Causal network 1: All observations
    parcorr = ParCorr(significance='analytic')
    dataframe1 = pp.DataFrame(df.values, datatime={0: np.arange(len(df))}, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe1, cond_ind_test=parcorr, verbosity=1)
    pcmci.verbosity = 1
    results1 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method='fdr_bh')

    # Causal network 2: Spring barrier

    # Construct mask to exclude spring barrier
    cycle_length = 12
    mask = np.ones(df.shape, dtype='bool')
    exc_spring = [0, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in exc_spring:
        mask[i::cycle_length, :] = False

    parcorr = ParCorr(significance='analytic', mask_type='y')
    dataframe2 = pp.DataFrame(np.copy(df), datatime={0: np.arange(len(df))}, var_names=var_names, mask=mask)
    pcmci = PCMCI(dataframe=dataframe2, cond_ind_test=parcorr, verbosity=1)
    results2 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method='fdr_bh')

    # Causal network 3: Spring barrier towards La Niña

    # range(start, stop, step)
    for s in range(11, int(len(df) / 12) * 12, 12):
        if np.mean(nino34[s - 1:s + 2]) > -0.5:
            mask[s - 7:s + 2] = True

    mask[:1] = True
    dataframe3 = pp.DataFrame(np.copy(df), datatime={0: np.arange(len(df))}, var_names=var_names, mask=mask)
    pcmci = PCMCI(dataframe=dataframe3, cond_ind_test=parcorr, verbosity=1)
    results3 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method='fdr_bh')

    # Causal network 4: Spring barrier towards El Niño
    mask = np.ones(df.shape, dtype='bool')
    exc_spring = [0, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in exc_spring:
        mask[i::cycle_length, :] = False

    for s in range(11, int(len(df) / 12) * 12, 12):
        if np.mean(nino34[s - 1:s + 2]) < 0.5:
            mask[s - 7:s + 2] = True

    mask[:1] = True
    dataframe4 = pp.DataFrame(np.copy(df), datatime={0: np.arange(len(df))}, var_names=var_names, mask=mask)
    pcmci = PCMCI(dataframe=dataframe4, cond_ind_test=parcorr, verbosity=1)
    results4 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method='fdr_bh')

    return results1, results2, results3, results4

def eww_coef_difference(reference_val_matrix, reference_p_matrix, val_matrix, p_matrix, threshold):
    mask_array = np.where(reference_p_matrix < threshold, 1, 0)
    mask_array2 = np.where(p_matrix < threshold, 1, 0)
    mask_array = mask_array.flatten().tolist()
    mask_array2 = mask_array2.flatten().tolist()
    eww_coefficients = []
    for s in range(59,80):
        eww_coefficients.append(s)

    for i in range(len(mask_array)):
        if i not in eww_coefficients:
          mask_array[i] = 0
          mask_array2[i] = 0
    reference_val_matrix = reference_val_matrix.flatten().tolist()
    val_matrix = val_matrix.flatten().tolist()
    eww_coef1 = []
    eww_coef2 = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1 and mask_array2[i] == 1:
            eww_coef1.append(reference_val_matrix[i])
            eww_coef2.append(val_matrix[i])

    return eww_coef1, eww_coef2




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

date = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS')
date = date.strftime('%Y-%m')


plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.plot(date, eww)
plt.xticks(rotation=45)
plt.plot(pp.smooth(np.copy(eww), smooth_width=15 * 12, residuals=False))
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.plot(eww.index, eww)
plt.plot(pp.smooth(np.copy(eww), smooth_width=15 * 12, residuals=False))
plt.xticks(rotation=45)
ax.set_ylabel('Speed ($m/s$)')
plt.title('Time series of Trade winds & long-term trend')




#start analysis

results1, results2, results3, results4 = run_pcmci(wpac, cpac, epac, nino34, eww, thf, start = '1950-01', end = '2025-01')
var_names = ['WPAC', 'CPAC', 'EPAC', 'EWW', 'THF']

results1['val_matrix']



#plot results
pink_patch = mpatches.Patch(color='pink', alpha = 0.4, label='WPAC, CPAC & EPAC regions')
plum_patch = mpatches.Patch(color='plum',alpha = 0.4, label='WWT & THF regions')

proj = ccrs.Orthographic(central_longitude=210.0, central_latitude=15.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)

#Add areas
x, y = [-180, -170, -160, -150, -150, -160, -170, -180], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='plum', alpha=0.4, zorder=0)
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, zorder=0)
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4,zorder=0)
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)

plt.legend(handles=[pink_patch, plum_patch], loc = 'lower center', fontsize = 'small')
# causal network plot

node_pos = {
    'y': np.array([-500000., -1000000., -500000., 2000000., -4000000.]),
    'x': np.array([-5500000., 1000000., 6000000., -2000000, -2000000])
}

ax = tp.plot_graph(
    fig_ax=(fig, ax),
    graph = results1['graph'],
    node_pos=node_pos,
    figsize=(10, 5),
    val_matrix=results1['val_matrix'],
    #edge_ticks=.5,
    var_names=var_names,
    arrow_linewidth= 4,
    curved_radius=.25,
    node_size=1000000.,
    #node_label_size=8,
    node_aspect= 1.,
    link_label_fontsize=7,
    #label_fontsize=8,
    node_colorbar_label='auto-MCI',
    link_colorbar_label='cross-MCI')
fig.suptitle("Causal network for EC-Earth3 model 1950-2025", fontsize=12)
plt.show()



#future networks

results1f, results2f, results3f, results4f = run_pcmci(wpac, cpac, epac, nino34, eww, thf, start = '2025-01', end = '2100-01')
var_names = ['WPAC', 'CPAC', 'EPAC', 'EWW', 'THF']


pink_patch = mpatches.Patch(color='pink', alpha = 0.4, label='WPAC, CPAC & EPAC regions')
plum_patch = mpatches.Patch(color='plum',alpha = 0.4, label='WWT & THF regions')

proj = ccrs.Orthographic(central_longitude=210.0, central_latitude=15.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas

x, y = [-180, -170, -160, -150, -150, -160, -170, -180], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='plum', alpha=0.4, zorder=0)
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, zorder=0)
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4,zorder=0)
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='pink', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)

plt.legend(handles=[pink_patch, plum_patch], loc = 'lower center', fontsize = 'small')
# causal network plot

node_pos = {
    'y': np.array([-500000., -1000000., -500000., 2000000., -4000000.]),
    'x': np.array([-5500000., 1000000., 6000000., -2000000, -2000000])
}

ax = tp.plot_graph(
    fig_ax=(fig, ax),
    graph = results4['graph'],
    node_pos=node_pos,
    figsize=(10, 5),
    val_matrix=results4['val_matrix'],
    #edge_ticks=.5,
    var_names=var_names,
    arrow_linewidth= 4,
    curved_radius=.25,
    node_size=1000000.,
    #node_label_size=8,
    node_aspect= 1.,
    link_label_fontsize=7,
    #label_fontsize=8,
    node_colorbar_label='auto-MCI',
    link_colorbar_label='cross-MCI')
fig.suptitle("Causal network for EC-Earth3 model 2025-2100", fontsize=12)
plt.show()



############################ Comparison past vs future networks ############################

link_difference(results1['val_matrix'], results1['p_matrix'], results1f['val_matrix'],results1f['p_matrix'], 0.05)
link_difference(results2['val_matrix'], results2['p_matrix'], results2f['val_matrix'],results2f['p_matrix'], 0.05)
link_difference(results3['val_matrix'], results3['p_matrix'], results3f['val_matrix'],results3f['p_matrix'], 0.05)
link_difference(results4['val_matrix'], results4['p_matrix'], results4f['val_matrix'],results4f['p_matrix'], 0.05)

comparison_metrics(results1['val_matrix'], results1['p_matrix'], results1f['val_matrix'], results1f['p_matrix'], 0.05)
comparison_metrics(results2['val_matrix'], results2['p_matrix'], results2f['val_matrix'], results2f['p_matrix'], 0.05)
comparison_metrics(results3['val_matrix'], results3['p_matrix'], results3f['val_matrix'], results3f['p_matrix'], 0.05)
comparison_metrics(results4['val_matrix'], results4['p_matrix'], results4f['val_matrix'], results4f['p_matrix'], 0.05)


############################ Trade wind analysis ############################


list1, list2 = eww_coef_difference(results1['val_matrix'], results1['p_matrix'], results1f['val_matrix'], results1f['p_matrix'], 0.05)
labels = [r'$WPAC_{t+3}$', r'$CPAC_{t+1}$', r'$CPAC_{t+2}$', r'$CPAC_{t+3}$', r'$EPAC_{t+1}$', r'$EWW_{t+1}$',
          r'$EWW_{t+2}$', r'$EWW_{t+3}$', r'$THF_{t+1}$', r'$THF_{t+2}$', r'$THF_{t+3}$']


# Set the width of the bars
bar_width = 0.35
# Create an array of indices for positioning the bars
indices = np.arange(len(list1))
# Plot the bars
plt.bar(indices, list1, bar_width, color='#2C5F2D', label='Network 1950-2025')
plt.bar(indices + bar_width, list2, bar_width, color='#97BC62FF', label='Network 2025-2100')
# Add labels, title, and legend
plt.xlabel('Data Points')
plt.ylabel('Partial correlation coef')
plt.title('Comparison of EWW Causal Strength')
plt.xticks(indices + bar_width/2, labels, rotation=70)
plt.legend()
plt.show()

for i in range(len(list1)):
    list1[i] = abs(list1[i])
    list2[i] = abs(list2[i])

np.mean(list1)
np.mean(list2)





############################ Robustness check ############################

var_names = ['tas_Amon', 'tas_Amon', 'psl_Amon', 'tos_Omon', 'uas_Amon', 'hfls_Amon', 'hfss_Amon']
model_names = ['HadGEM3-GC31-LL', 'MIROC6', 'ACCESS-CM2', 'GFDL-ESM4', 'MPI-ESM1-2-LR', 'FGOALS-f3-L']
scenario_time_period = ['historical_19500116-20141216.nc', 'ssp585_20150116-21001216.nc']
scenario_time_period_cpac = ['historical_19500116-20141216_2.nc', 'ssp585_20150116-21001216_2.nc']

comparison_metrics_model1 = []
comparison_metrics_model2 = []
comparison_metrics_model3 = []
comparison_metrics_model4 = []

model = 'HadGEM3-GC31-LL'

for model in model_names:
    epac_past = ncfiles_to_series(str(model) + '/' + str(var_names[1]) + '_' + str(model) + '_' + scenario_time_period[0],
        'tas', time_period="Past")
    epac_future = ncfiles_to_series(str(model) + '/' + str(var_names[1]) + '_' + str(model) + '_' + scenario_time_period[1],
        'tas', time_period="Future")
    epac = epac_past.append(epac_future)

    cpac_past = ncfiles_to_series(str(model) + '/' + str(var_names[1]) + '_' + str(model) + '_' + scenario_time_period_cpac[0],
                                  'tas', time_period="Past")
    cpac_future = ncfiles_to_series(str(model) + '/' + str(var_names[1]) + '_' + str(model) + '_' + scenario_time_period_cpac[1],
                                    'tas', time_period="Future")
    cpac = cpac_past.append(cpac_future)

    wpac_past = ncfiles_to_series(str(model) + '/' + str(var_names[2]) + '_' + str(model) + '_' + scenario_time_period[0],
                                  'psl', time_period="Past")
    wpac_future = ncfiles_to_series(str(model) + '/' + str(var_names[2]) + '_' + str(model) + '_' + scenario_time_period[1],
                                    'psl', time_period="Future")
    wpac = wpac_past.append(wpac_future)

    nino34_past = ncfiles_to_series(str(model) + '/' + str(var_names[3]) + '_' + str(model) + '_' + scenario_time_period[0],
                                  'tos', time_period="Past")
    nino34_future = ncfiles_to_series(str(model) + '/' + str(var_names[3]) + '_' + str(model) + '_' + scenario_time_period[1],
                                    'tos', time_period="Future")
    nino34 = nino34_past.append(nino34_future)

    eww_past = ncfiles_to_series(str(model) + '/' + str(var_names[4]) + '_' + str(model) + '_' + scenario_time_period[0],
                                    'uas', time_period="Past")
    eww_future = ncfiles_to_series(str(model) + '/' + str(var_names[4]) + '_' + str(model) + '_' + scenario_time_period[1],
                                      'uas', time_period="Future")
    eww = eww_past.append(eww_future)

    lhf_past = ncfiles_to_series(str(model) + '/' + str(var_names[5]) + '_' + str(model) + '_' + scenario_time_period[0],
                                 'hfls', time_period="Past")
    lhf_future = ncfiles_to_series(str(model) + '/' + str(var_names[5]) + '_' + str(model) + '_' + scenario_time_period[1],
                                   'hfls', time_period="Future")
    lhf = lhf_past.append(lhf_future)

    shf_past = ncfiles_to_series(str(model) + '/' + str(var_names[6]) + '_' + str(model) + '_' + scenario_time_period[0],
                                 'hfss', time_period="Past")
    shf_future = ncfiles_to_series(str(model) + '/' + str(var_names[6]) + '_' + str(model) + '_' + scenario_time_period[1],
                                   'hfss', time_period="Future")
    shf = shf_past.append(shf_future)

    eww = eww * -1
    thf = lhf + shf

    results1other, results2other, results3other, results4other = run_pcmci(wpac, cpac, epac, nino34, eww, thf, start='2025-01',
                                                       end='2100-01')

    comparison_metrics_model1.append(
        comparison_metrics(results1['val_matrix'], results1['p_matrix'], results1other['val_matrix'],
                           results1other['p_matrix'], 0.05))
    comparison_metrics_model2.append(
        comparison_metrics(results2['val_matrix'], results2['p_matrix'], results2other['val_matrix'],
                           results2other['p_matrix'], 0.05))
    comparison_metrics_model3.append(
        comparison_metrics(results3['val_matrix'], results3['p_matrix'], results3other['val_matrix'],
                           results3other['p_matrix'], 0.05))
    comparison_metrics_model4.append(
        comparison_metrics(results4['val_matrix'], results4['p_matrix'], results4other['val_matrix'],
                           results4other['p_matrix'], 0.05))



f1_scores = []
for i in range(len(model_names)):
    f1 = comparison_metrics_model1[i][4]
    f1_scores.append(f1)

for i in range(len(model_names)):
    f1 = comparison_metrics_model2[i][4]
    f1_scores.append(f1)

for i in range(len(model_names)):
    f1 = comparison_metrics_model3[i][4]
    f1_scores.append(f1)

for i in range(len(model_names)):
    f1 = comparison_metrics_model4[i][4]
    f1_scores.append(f1)


data_box_plot = pd.DataFrame({'Value': f1_scores,
                     'Category': ['Model 1'] * 6 + ['Model 2'] * 6 +
                                 ['Model 3'] * 6 + ['Model 4'] * 6})

# Create a figure and axis
fig, ax = plt.subplots()

# Create a dictionary to map categories to colors
colors = {'Model 1': '#2E5266FF', 'Model 2': '#6E8898FF', 'Model 3': '#9FB1BCFF', 'Model 4': '#D3D0CBFF'}
# Create a list to store the data for each category
category_data = []
# Iterate over each category and extract the values
for category, group in data_box_plot.groupby('Category'):
    category_data.append(group['Value'].values)
# Create the box plots
box = ax.boxplot(category_data, patch_artist=True, whis = 10)
# Set box plot colors
for patch, color in zip(box['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
# Add individual observations as points
for i, (category, group) in enumerate(data_box_plot.groupby('Category')):
    positions = np.random.normal(i + 1, 0.000001, size=len(group))
    ax.scatter(positions, group['Value'], color=colors[category], alpha=0.9)
# Set the x-axis tick labels
ax.set_xticklabels(colors.keys())
# Set the y-axis label
ax.set_ylabel(r'$F_1$ score')
# Set the plot title
ax.set_title('Inter-model variability: Comparison of CMIP6 model networks \n to EC-Earth3 network (2025-2100)' )
# Show the plot
plt.show()






data_box_plot = pd.DataFrame({'Value': f1_scores, 'Category': model_names * 4})
# Group the DataFrame by the 'Category' column and reset the index
data_box_plot = data_box_plot .groupby('Category').apply(lambda x: x.reset_index(drop=True))
# Reset the index of the DataFrame
data_box_plot = data_box_plot.reset_index(drop=True)
model_names = ['ACCESS-CM2', 'FGOALS-f3-L', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'MIROC6', 'MPI-ESM1-2-LR']
colors = ["#EA5545", "#F46A9B", "#EF9B20", "#EDBF33", "#EDE15B", "#BDCF32"]

fig, ax = plt.subplots()

# Create a dictionary to map categories to colors
colors = dict(zip(model_names, colors))

# Create a list to store the data for each category
category_data = []
# Iterate over each category and extract the values
for category, group in data_box_plot.groupby('Category'):
    category_data.append(group['Value'].values)

# Create the box plots with horizontal orientation
box = ax.boxplot(category_data, patch_artist=True, whis=10, vert=False)

# Set box plot colors
for patch, color in zip(box['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

# Add individual observations as points
for i, (category, group) in enumerate(data_box_plot.groupby('Category')):
    positions = np.random.normal(i + 1, 0.000001, size=len(group))
    ax.scatter(group['Value'], positions, color=colors[category], alpha=0.9)
# Set the y-axis tick labels
ax.set_yticklabels(colors.keys())
# Set the x-axis label
ax.set_xlabel(r'$F_1$ score')
# Set the plot title
ax.set_title('Inter-model variability: Comparison of CMIP6 model networks \n to EC-Earth3 network (2025-2100)')

# Show the plot
plt.show()
