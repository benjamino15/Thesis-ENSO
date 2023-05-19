import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
import random
import statsmodels.api as sm

from tigramite.independence_tests.parcorr import ParCorr

x = np.random.normal(0, 1, 1000)
y = np.random.normal(5, 0.5, 1000)
z = np.random.normal(-5, 0.5, 1000)

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.show()

df = pd.DataFrame({'x':x, 'y':y, 'z':z})
var_names = ['x', 'y', 'z']
dataframe = pp.DataFrame(df.values, datatime = {0:np.arange(len(df))}, var_names= var_names)

def t_transform(x, y):
    r = np.corrcoef(x, y)
    r = r[0,1]
    t = r * ((len(x)/(1-r**2))**0.5)
    return t

z = sm.add_constant(z)
model1 = sm.OLS(x,z)
results1 = model1.fit()
r1 = results1.resid

def sim_parCorr(T, t_crit):

    reject_list = []

    for i in range(T):
        random.seed(i)
        x = np.random.normal(0, 1, T)
        y = np.random.normal(5, 0.5, T)
        z = np.random.normal(-5, 0.5, T)

        # OLS
        z = sm.add_constant(z)
        model1 = sm.OLS(x, z)
        results1 = model1.fit()
        r1 = results1.resid

        model2 = sm.OLS(y, z)
        results2 = model2.fit()
        r2 = results2.resid

        # t transformation
        t = t_transform(r1, r2)
        # 99% critical value for T = 1000
        if t>t_crit or t< -t_crit:
            reject_list.append(1)
        else:
            reject_list.append(0)

    return np.mean(reject_list)

sim_parCorr(1000, 2.581)


false_positives = []
for i in range(10):
    iteration = i
    false_positive = sim_parCorr(1000, 2.581)
    false_positives.append(false_positive)


