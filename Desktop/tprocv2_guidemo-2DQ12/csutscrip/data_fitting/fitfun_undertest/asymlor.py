import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
from fitting import *


np.bool= bool
np.float = np.float64
res = r'Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY(entry=2)


def asym_lorfunc(x, y0, A, x0, gamma, alpha):
    return y0 + A / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0))))**2)


def fit_asym_lor(xdata, ydata, fitparams=None):
    if fitparams is None: 
        fitparams = [None] * 5
    else: 
        fitparams = np.copy(fitparams)

    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None: fitparams[4] = 0

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(asym_lorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        print('Warning: fit failed!')
    return pOpt, pCov


frequencies = np.linspace(4.0, 5.0, 1000)
asym_lorentzian_dip = -1.0 / (1 + ((frequencies - 4.5) / (0.01 * (1 + -30 * (frequencies - 4.5))))**2)
noise = np.random.normal(0, 0.02, frequencies.shape)
noisy_asym_lorentzian_dip = asym_lorentzian_dip + noise
plt.plot(frequencies, asym_lorentzian_dip)
plt.show()



# y = np.abs(y)
# pOpt, pCov = fit_asym_lor(x, y)
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='Noisy Asymmetric Lorentzian Dip', color='black', linestyle='-', alpha=0.7)
# plt.plot(x, asym_lorfunc(x, *pOpt), label='Fitted Asymmetric Lorentzian', color='red', linestyle='--')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Amplitude')
# print('asym=', pOpt[4])
# plt.axvline(pOpt[2])
# plt.title(f'Noisy Asymmetric Lorentzian Dip at {(pOpt[2]*1e-9):.2f}GHz with Fitting')
# plt.legend()
# plt.grid(True)
# plt.show()
