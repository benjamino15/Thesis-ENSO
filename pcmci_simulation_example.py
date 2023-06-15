import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from numpy.random import multivariate_normal

def simulate_dependent_autocorrelated_variables(n, rho, sigma):
    # Generate random noise
    np.random.seed(123)
    epsilon = np.random.multivariate_normal([0, 0], [[sigma**2, rho*sigma**2], [rho*sigma**2, sigma**2]], n)

    # Initialize the variables
    x = np.zeros(n)
    y = np.zeros(n)

    # Generate the values for the variables
    for i in range(1, n):
        x[i] = rho * x[i-1] + epsilon[i, 0]
        y[i] = rho * y[i-1] + epsilon[i, 1]

    return x, y




# Simulation parameters
n = 1000  # Number of data points
rho = 0.8  # Autocorrelation coefficient
sigma = 1.0  # Standard deviation

# Simulate dependent autocorrelated variables
x, y = simulate_dependent_autocorrelated_variables(n, rho, sigma)

plt.scatter(x, y)
plt.show()

plt.plot(x)
plt.plot(y)
plt.show()

df = pd.DataFrame({'x':x, 'y':y})
var_names = ['X', 'Y']

x_lag = df['x'].shift(-1)
x_lag = x_lag.dropna()

np.corrcoef(df['x'][:-1], x_lag)


parcorr = ParCorr(significance='analytic', verbosity=3)
dataframe1 = pp.DataFrame(df.values, datatime={0: np.arange(len(df))}, var_names=var_names)
pcmci = PCMCI(dataframe=dataframe1, cond_ind_test=parcorr, verbosity=4)

results1 = pcmci.run_pcmci(tau_max=2, tau_min=1, pc_alpha=0.05, alpha_level=0.05, fdr_method='fdr_bh')
