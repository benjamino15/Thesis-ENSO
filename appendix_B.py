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


epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.0_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)

var_names = ['WPAC', 'CPAC', 'EPAC']
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')

def plot_pcmci(results1, results2, results3, results4, var_names, add_to_title = "ensemble run 0", year = '1950-2025'):
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

    fig.suptitle("Causal networks for EC-Earth3 model " + year + add_to_title, fontsize=12)
    plt.subplots_adjust(hspace=0.4)
    plt.show()



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.1_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 1", year = '1950-2025')

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.1_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.1_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 1", year = '2025-2100')



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.2_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 2", year = '1950-2025')


epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.2_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.2_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 2", year = '2025-2100')



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.3_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 3", year = '1950-2025')

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.3_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.3_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 3", year = '2025-2100')



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.4_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 4", year = '1950-2025')

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.4_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.4_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 4", year = '2025-2100')



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.5_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 5", year = '1950-2025')

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.5_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.5_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 5", year = '2025-2100')



epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.6_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '1950-01', end = '2025-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 6", year = '1950-2025')

epac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-100--80E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
cpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-150--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
wpac = pd.read_csv('CMIP6/EC-Earth3/icmip6_psl_mon_EC-Earth3_ssp585.6_130-150E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
nino34 = pd.read_csv('CMIP6/EC-Earth3/icmip6_tas_mon_EC-Earth3_ssp585.6_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=155, header=None)
results1, results2, results3, results4 = preprocessing_run_pcmci(wpac, cpac, epac, nino34, start = '2025-01', end = '2100-01')
plot_pcmci(results1, results2, results3, results4, var_names, add_to_title=", ensemble run 6", year = '2025-2100')