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
    auto_coefficients = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35]
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
            difference.append(reference_val_matrix[i])
        if mask_array2[i] == 1:
            difference2.append(val_matrix[i])
    return (np.mean(difference) - np.mean(difference2))
def preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01'):
    start_year_data = epac[0][0]
    cpac.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
    date = pd.date_range(start=str(start_year_data)+'-01-01', end='2100-12-01', freq='MS')
    date = date.strftime('%Y-%m')

    df = pd.DataFrame({'Date': date, 'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})
    df = df[df['Date'] <= end]
    df = df[df['Date'] >= start]

    var_names = ['WPAC', 'CPAC', 'EPAC']
    cpac = deTrend_deSeasonalize(df['CPAC'], show_plot=False)
    epac = deTrend_deSeasonalize(df['EPAC'], show_plot=False)
    wpac = deTrend_deSeasonalize(df['WPAC'], show_plot=False)

    df_nino34 = pd.DataFrame({'Date': date, 'nino34': nino34})
    df_nino34 = df_nino34[df_nino34['Date'] <= end]
    df_nino34 = df_nino34[df_nino34['Date'] >= start]
    nino34 = df_nino34['nino34']
    #detrend nino34 index
    nino34 = pp.smooth(np.copy(nino34), smooth_width=30 * 12, residuals=True)

    #start analysis
    df = pd.DataFrame({'WPAC': wpac, 'CPAC': cpac, 'EPAC': epac})

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

##############################################################################################
epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.0_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)

results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
var_names = ['WPAC', 'CPAC', 'EPAC']

# Overall graph plot

fig, axes = plt.subplots(2, 2)

tp.plot_graph(
    val_matrix=results1['val_matrix'],
    graph=results1['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    fig_ax= (fig, axes[0,0]))
axes[0, 0].set_title('All observations', fontsize=10)

tp.plot_graph(
    val_matrix=results2['val_matrix'],
    graph=results2['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[0,1]))
axes[0, 1].set_title('Spring barrier', fontsize=10)

tp.plot_graph(
    val_matrix=results3['val_matrix'],
    graph=results3['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,0]))
axes[1, 0].set_title('Spring barrier towards La Niña', fontsize=10)

tp.plot_graph(
    val_matrix=results4['val_matrix'],
    graph=results4['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,1]))
axes[1, 1].set_title('Spring barrier towards El Niño', fontsize=10)

fig.suptitle("Causal networks for EC-Earth3 model 1950-2025", fontsize=16)
plt.subplots_adjust(hspace=0.4)
plt.show()

results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')

var_names = ['WPAC', 'CPAC', 'EPAC']
# Overall graph plot

fig, axes = plt.subplots(2, 2)

tp.plot_graph(
    val_matrix=results1['val_matrix'],
    graph=results1['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    fig_ax= (fig, axes[0,0]))
axes[0, 0].set_title('All observations', fontsize=10)

tp.plot_graph(
    val_matrix=results2['val_matrix'],
    graph=results2['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[0,1]))
axes[0, 1].set_title('Spring barrier', fontsize=10)

tp.plot_graph(
    val_matrix=results3['val_matrix'],
    graph=results3['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,0]))
axes[1, 0].set_title('Spring barrier towards La Niña', fontsize=10)

tp.plot_graph(
    val_matrix=results4['val_matrix'],
    graph=results4['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    fig_ax= (fig, axes[1,1]))
axes[1, 1].set_title('Spring barrier towards El Niño', fontsize=10)

fig.suptitle("Causal networks for EC-Earth3 model 2025-2100", fontsize=16)
plt.subplots_adjust(hspace=0.4)
plt.show()

#Comparison walker circulation plots past vs future
epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.0_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)

results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
results1f, results2f, results3f, results4f = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
var_names = ['WPAC', 'CPAC', 'EPAC']

link_difference(results1['val_matrix'], results1['p_matrix'], results1f['val_matrix'],results1f['p_matrix'], 0.05)
link_difference(results2['val_matrix'], results2['p_matrix'], results2f['val_matrix'],results2f['p_matrix'], 0.05)
link_difference(results3['val_matrix'], results3['p_matrix'], results3f['val_matrix'],results3f['p_matrix'], 0.05)
link_difference(results4['val_matrix'], results4['p_matrix'], results4f['val_matrix'],results4f['p_matrix'], 0.05)

comparison_metrics(results1['val_matrix'], results1['p_matrix'], results1f['val_matrix'], results1f['p_matrix'], 0.05)
comparison_metrics(results2['val_matrix'], results2['p_matrix'], results2f['val_matrix'], results2f['p_matrix'], 0.05)
comparison_metrics(results3['val_matrix'], results3['p_matrix'], results3f['val_matrix'], results3f['p_matrix'], 0.05)
comparison_metrics(results4['val_matrix'], results4['p_matrix'], results4f['val_matrix'], results4f['p_matrix'], 0.05)

#comparison with other networks

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.0_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)



ensemble_number = [1, 2, 3, 4, 5, 6]
threshold = 0.05

comparison_metrics_model1 = []
comparison_metrics_model2 = []
comparison_metrics_model3 = []
comparison_metrics_model4 = []


for i in range(len(ensemble_number)):
    epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.' + str(ensemble_number[i]) + '_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+",
                       skiprows=120, header=None)
    cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.' + str(ensemble_number[i]) + '_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+",
                       skiprows=120, header=None)
    wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.' + str(ensemble_number[i]) + '_130-150E_-5-5N_n_su.dat', delimiter=r"\s+",
                       skiprows=120, header=None)
    nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.' + str(ensemble_number[i]) + '_-170--120E_-5-5N_n_su.dat',delimiter=r"\s+",
                       skiprows=120, header=None)

    results1other, results2other, results3other, results4other = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start='2025-01',end='2100-01')

    comparison_metrics_model1.append(comparison_metrics(results1['val_matrix'], results1['p_matrix'], results1other['val_matrix'], results1other['p_matrix'],threshold))
    comparison_metrics_model2.append(comparison_metrics(results2['val_matrix'], results2['p_matrix'], results2other['val_matrix'], results2other['p_matrix'],threshold))
    comparison_metrics_model3.append(comparison_metrics(results3['val_matrix'], results3['p_matrix'], results3other['val_matrix'], results3other['p_matrix'],threshold))
    comparison_metrics_model4.append(comparison_metrics(results4['val_matrix'], results4['p_matrix'], results4other['val_matrix'], results4other['p_matrix'],threshold))

f1_scores = []
for i in range(len(ensemble_number)):
    f1 = comparison_metrics_model1[i][4]
    f1_scores.append(f1)

for i in range(len(ensemble_number)):
    f1 = comparison_metrics_model2[i][4]
    f1_scores.append(f1)

for i in range(len(ensemble_number)):
    f1 = comparison_metrics_model3[i][4]
    f1_scores.append(f1)

for i in range(len(ensemble_number)):
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
box = ax.boxplot(category_data, patch_artist=True)
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
ax.set_title('Intra-model variability: Comparison of ensemble runs \n to the benchmark network (2025-2100)' )
# Show the plot
plt.show()