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


# to check tigramite version
print(pkg_resources.get_distribution("tigramite").version)

# Reading the data
epac = pd.read_csv('CMIP6/CESM2/icmip6_tas_mon_CESM2_ssp585.0_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=156, header=None)
cpac = pd.read_csv('CMIP6/CESM2/icmip6_tas_mon_CESM2_ssp585.0_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=156, header=None)
wpac = pd.read_csv('CMIP6/CESM2/icmip6_psl_mon_CESM2_ssp585.0_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=156, header=None)
nino34 = pd.read_csv('CMIP6/CESM2/icmip6_tas_mon_CESM2_ssp585.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=156, header=None)


cpac.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
epac.columns = cpac.columns
wpac.columns = cpac.columns
nino34.columns = cpac.columns


# stack observations into a single column
cpac = cpac.drop(cpac.columns[0], axis=1)
cpac = cpac.stack().reset_index(drop=True)

epac = epac.drop(epac.columns[0], axis=1)
epac = epac.stack().reset_index(drop=True)

wpac = wpac.drop(wpac.columns[0], axis=1)
wpac = wpac.stack().reset_index(drop=True)

nino34 = nino34.drop(nino34.columns[0], axis=1)
nino34 = nino34.stack().reset_index(drop=True)


# specify date and add to dataframe
date = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS')
date = date.strftime('%Y-%m')

df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})
df = df[df['Date'] <= '2025-01']
df = df[df['Date'] >= '1950-01']




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




#we use the El niño 3.4 index to mask the data, so we can choose only the data during the Nov-Feb period with La Niña-Neutral conditions
#Oceanic Nino Index defined by 5 consecutive months of 3-month-running-mean of Nino3.4 SST above/below 0.5

df_nino34 = pd.DataFrame({'Date': date, 'nino34': nino34})
df_nino34 = df_nino34[df_nino34['Date'] <= '2025-01']
df_nino34 = df_nino34[df_nino34['Date'] >= '1950-01']
nino34 = df_nino34['nino34']

plt.plot(pp.smooth(np.copy(nino34), smooth_width=30 * 12, residuals=False))
plt.plot(nino34)
plt.show()

nino34 = pp.smooth(np.copy(nino34), smooth_width=30 * 12, residuals=True)
plt.plot(nino34)
plt.show()


# to see averages
test = []
for s in range(11, int(len(nino34)/12)*12, 12):
    test.append(np.mean(nino34[s-1:s+2]))
enso_state = []
for i in range(len(test)):
    if test[i] < -0.5:
        enso_state.append('A')
    elif test[i] > 0.5:
        enso_state.append('O')
    else:
        enso_state.append('N')

enso_state.count('A')
enso_state.count('O')
enso_state.count('N')

enso_state_year = pd.DataFrame({'enso_state': enso_state, 'year': range(1950, 2025)})

# start analysis

df = pd.DataFrame({'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})

# create tigramite dataframe to inspect
dataframe = pp.DataFrame(df.values, datatime = {0:np.arange(len(df))}, var_names= var_names)

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







# Causal network 1: All observations
parcorr = ParCorr(significance='analytic')
dataframe1 = pp.DataFrame(df.values, datatime = {0:np.arange(len(df))}, var_names= var_names)
pcmci.verbosity = 1
pcmci = PCMCI(dataframe=dataframe1, cond_ind_test=parcorr,verbosity=1)
results1 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha= 0.05, alpha_level=0.05, fdr_method = 'fdr_bh')


# array shape n x n x m. n variables (matrices and rows), m lags (columns).
# matrix i represents the effect variable i has on other variables.
# for example, matrix i, row j, column k, represents the effect of variable i on variable j, in lag k-1.
array = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

threshold = 0.05

reference_val_matrix = results1['val_matrix']
reference_p_matrix = results1['p_matrix']
val_matrix = results2['val_matrix']
p_matrix = results2['p_matrix']

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
    return np.mean(abs_difference), np.sum(squared_difference)**0.5, false_negatives, false_positives, f1_score



comparison_metrics(results1['val_matrix'], results1['p_matrix'], results2['val_matrix'],results2['p_matrix'], threshold)





mask_array = np.where(results2['p_matrix'] < threshold, 1, 0)
mask_array
mask_array.flatten().tolist()

# Causal network 2: Spring barrier

# Construct mask to exclude spring barrier
cycle_length = 12
mask = np.ones(df.shape, dtype='bool')
exc_spring = [0, 4, 5, 6, 7, 8, 9, 10, 11]
for i in exc_spring:
    mask[i::cycle_length, :] = False

parcorr = ParCorr(significance='analytic', mask_type='y')
dataframe2 = pp.DataFrame(np.copy(df), datatime = {0:np.arange(len(df))}, var_names= var_names, mask = mask)
pcmci = PCMCI(dataframe=dataframe2, cond_ind_test=parcorr,verbosity=1)
results2 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha= 0.05, alpha_level=0.05, fdr_method = 'fdr_bh')

results2['val_matrix']

# Causal network 3: Spring barrier towards La Niña

# range(start, stop, step)
for s in range(11, int(len(df)/12)*12, 12):
  if np.mean(nino34[s-1:s+2]) >-0.5:
      mask[s-7:s+2] = True

for s in range(11, int(len(df)/12)*12, 12):
    print(s)

#nina_mask = np.ones(length, dtype = 'bool')
#for t in range(length):
#    if np.sum(nino34smoothed[max(0, t-2): min(length, t+3)] < -0.5) >= 5:
#        nina_mask[t] = False

#for t in range(mask.shape[0]):
#   if nina_mask[t] == True:
#        mask[t] = True

dataframe3 = pp.DataFrame(np.copy(df), datatime = {0:np.arange(len(df))}, var_names= var_names, mask = mask)
pcmci = PCMCI(dataframe=dataframe3, cond_ind_test=parcorr,verbosity=1)
results3 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha= 0.05, alpha_level=0.05, fdr_method = 'fdr_bh')


# Causal network 4: Spring barrier towards El Niño
mask = np.ones(df.shape, dtype='bool')
exc_spring = [0, 4, 5, 6, 7, 8, 9, 10, 11]
for i in exc_spring:
    mask[i::cycle_length, :] = False

for s in range(11, int(len(df)/12)*12, 12):
  if np.mean(nino34[s-1:s+2]) < 0.5:
      mask[s-7:s+2] = True

#nino_mask = np.ones(length, dtype = 'bool')
#for t in range(length):
#    if np.sum(nino34smoothed[max(0, t-2): min(length, t+3)] > 0.5) >= 5:
#        nino_mask[t] = False

#for t in range(mask.shape[0]):
#   if nino_mask[t] == True:
#        mask[t] = True

dataframe4 = pp.DataFrame(np.copy(df), datatime = {0:np.arange(len(df))}, var_names= var_names, mask = mask)
pcmci = PCMCI(dataframe=dataframe4, cond_ind_test=parcorr,verbosity=1)
results4 = pcmci.run_pcmci(tau_max=3, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method = 'fdr_bh')

tp.plot_timeseries(dataframe, color='black', show_meanline=True, grey_masked_samples='data')
plt.show()


# Overall graph plot

fig, axes = plt.subplots(2, 2)

tp.plot_graph(
    val_matrix=results1['val_matrix'],
    graph=results1['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    fig_ax= (fig, axes[0,0]))
axes[0, 0].set_title('All observations')

tp.plot_graph(
    val_matrix=results2['val_matrix'],
    graph=results2['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[0,1]))
axes[0, 1].set_title('Spring barrier')

tp.plot_graph(
    val_matrix=results3['val_matrix'],
    graph=results3['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,0]))
axes[1, 0].set_title('Spring barrier towards La Niña')

tp.plot_graph(
    val_matrix=results4['val_matrix'],
    graph=results4['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,1]))
axes[1, 1].set_title('Spring barrier towards El Niño')

fig.suptitle("Causal networks for CMIP6 data", fontsize=16)
plt.subplots_adjust(hspace=0.4)
plt.show()
