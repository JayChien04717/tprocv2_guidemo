import matplotlib.pyplot as plt
import numpy as np
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\5q data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
from lmfit import create_params, minimize
np.bool= bool
np.float = np.float64
# res = r'./r1_cal.hdf5'
res = r'./5Q_onetone_pdr_6.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY()

from lmfit import Model
import lmfit

def hangerfunc(x, f0, Qi, Qe, phi, scale, a0):
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

def hangerS21func(x, f0, Qi, Qe, phi, scale, a0):
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return a0 + np.abs(hangerfunc(x, f0, Qi, Qe, phi, scale, a0)) - scale*(1-Q0/Qe)

def hangerS21func_sloped(x, f0, Qi, Qe, phi, scale, a0, slope):
    return hangerS21func(x, f0, Qi, Qe, phi, scale, a0) + slope*(x-f0)


def fithanger(xdata, ydata):
    params = lmfit.create_params(
    f0 = dict(value = np.average(xdata), min =xdata[0], max = xdata[-1]),
    Qi = dict(value = 10000, min = 1000, max = 1e8),
    Qe = dict(value = 5000, min = 1000, max= 1e8),
    phi = dict(value = 0, min = -np.pi, max = np.pi),
    a0 = dict(value=np.average(ydata), min = np.min(ydata), max = np.max(ydata)),
    slope = dict(value=(ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])),
    )




y = np.abs(y)
mod = Model(hangerS21func_sloped)
pars = mod.make_params(
    f0={'value':np.average(x), 'min': x[0], 'max':x[-1]},
    Qi={'value': 10000, 'min':1000, 'max':1e6}, 
    Qe={'value': 1000, 'min': 1000, 'max':1e6}, 
    phi={'value': 0, 'min':-np.pi, 'max':np.pi}, 
    scale={'value': max(y)-min(y), 'min':min(y), 'max': max(y)},
    a0={'value': np.average(y), 'min':min(y), 'max': max(y)}, 
    slope={'value': ((y[-1] - y[0]) / (x[-1] - x[0])), 'min':-100, 'max': 100}
)
result = mod.fit(data=y, params=pars, x=x, method='least_squares')
dely = result.eval_uncertainty(sigma=3)
print(result.fit_report(min_correl=0.5))

plt.plot(x, y)
plt.plot(x, result.best_fit, '-', label='best fit')
plt.fill_between(x, result.best_fit-dely, result.best_fit+dely,
                 color="#ABABAB", label=r'3-$\sigma$ uncertainty band')
plt.legend()
plt.show()