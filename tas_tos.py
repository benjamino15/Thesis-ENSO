import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tigramite import data_processing as pp


tas = pd.read_csv('CMIP5/icmip5_tas_Amon_modmean_rcp85.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=120, header=None)
tos = pd.read_csv('CMIP5/icmip5_tos_Omon_modmean_rcp85.0_-170--120E_-5-5N_n_su.dat', delimiter=r"\s+", skiprows=119, header=None)

tas.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
tos.columns = tos.columns

tas = tas.drop(tas.columns[0], axis=1)
tas = tas.stack().reset_index(drop=True)

tos = tos.drop(tos.columns[0], axis=1)
tos = tos.stack().reset_index(drop=True)

date = pd.date_range(start='1950-01-01', end='2100-12-01', freq='MS')
date = date.strftime('%Y-%m')

df = pd.DataFrame({'Date': date, 'tas': tas, 'tos': tos})
df = df[df['Date'] <= '2023-03']
df = df[df['Date'] >= '1950-01']

tas = df['tas']
tos = df['tos']


tas = pp.smooth(np.copy(tas), smooth_width=30 * 12, residuals=True)
tos = pp.smooth(np.copy(tos), smooth_width=30 * 12, residuals=True)



plt.plot(tas)
plt.plot(tos)
plt.show()

tas = df['tas']
    .shift(0, axis = 0)
tas = tas[:-0]

np.corrcoef(df['tos'], tas)









