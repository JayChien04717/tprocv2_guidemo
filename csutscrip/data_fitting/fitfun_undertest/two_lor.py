import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
from scipy.signal import find_peaks
np.bool= bool
np.float = np.float64
res = r'C:\Users\QEL\Desktop\fit_pack\data\twotone\Normalized_chalmers@4.7676_EC fine.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY(entry=16)


def lorfunc(x, *p):
    y0, yscale1, f0, gamma1, yscale2, f1, gamma2 = p
    return y0 + (yscale1 / (1 + (x - f0)**2 / gamma1**2)) + (yscale2 / (1 + (x - f1)**2 / gamma2**2))


def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None: 
        fitparams = [None] * 7
    else: 
        fitparams = np.copy(fitparams)

    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None: fitparams[4] = (max(ydata) - min(ydata)) /50
    if fitparams[5] is None: fitparams[5] =  xdata[np.argmax(abs(ydata - fitparams[0])-20)]  
    if fitparams[6] is None: fitparams[6] = (max(xdata) - min(xdata)) / 10

    fitparams = [0 if param is None else param for param in fitparams]

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        print('Warning: fit failed!')
    return pOpt, pCov


# frequencies = np.linspace(4.0, 5.0, 1000)
# lorentzian_dip_1 = 1.0 / (1 + (frequencies - 4.5)**2 / 0.05**2)
# lorentzian_dip_2 = 1.0 / (1 + (frequencies - 4.2)**2 / 0.05**2)
# combined_lorentzian_dip = lorentzian_dip_1 + lorentzian_dip_2
# noise = np.random.normal(0, 0.02, frequencies.shape)
# noisy_combined_lorentzian_dip = combined_lorentzian_dip + noise

y = np.abs(y)
pOpt, pCov = fitlor(x, y)
print(x, y)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Noisy Combined Lorentzian Dip', color='black', linestyle='-', alpha=0.7)
plt.plot(x, lorfunc(x, *pOpt), label='Fitted Lorentzian', color='red', linestyle='--')
# plt.axvspan(pOpt[2] - pOpt[3], pOpt[2] + pOpt[3], alpha=0.2, label=f'BW={(pOpt[3]*1e-6):.2f} MHz')
# plt.axvspan(pOpt[5] - pOpt[6], pOpt[5] + pOpt[6], alpha=0.2, label=f'BW={(pOpt[6]*1e-6):.2f} MHz')
# plt.axvline(pOpt[2])
# plt.axvline(pOpt[5])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()





