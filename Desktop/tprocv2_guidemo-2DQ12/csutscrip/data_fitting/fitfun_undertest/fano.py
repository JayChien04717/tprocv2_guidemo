import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber


np.bool= bool
np.float = np.float64
res = r'Normalized_coupler_chen_054[7]_@7.819GHz_power_dp_002.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY(entry=3)
# Fano 函数
def fano_func(x, y0, A, x0, gamma, q):
    return y0 + A * ((q * gamma + x - x0)**2) / (gamma**2 + (x - x0)**2)

# 拟合函数
def fit_fano(xdata, ydata, fitparams=None):
    if fitparams is None: 
        fitparams = [None] * 5
    else: 
        fitparams = np.copy(fitparams)

    if fitparams[0] is None: 
        fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: 
        fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: 
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: 
        fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None: 
        fitparams[4] = 1 
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.curve_fit(fano_func, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        print('Warning: fit failed!')
    return pOpt, pCov


# frequencies = np.linspace(4.0, 5.0, 1000)
# fano_resonance = 1 + ((-0.5 * 0.05 + frequencies - 4.5)**2) / (0.05**2 + (frequencies - 4.5)**2)
# noise = np.random.normal(0, 0.02, frequencies.shape)
# noisy_fano_resonance = fano_resonance + noise

y = np.abs(y)

pOpt, pCov = fit_fano(x, y)


plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Noisy Fano Resonance', color='black', linestyle='-', alpha=0.7)
plt.plot(x, fano_func(x, *pOpt), label='Fitted Fano Resonance', color='red', linestyle='--')
# plt.axvline(pOpt[2])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
