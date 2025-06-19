import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
np.bool= bool
np.float = np.float64
res = r'C:\Users\QEL\Desktop\fit_pack\data\twotone\Normalized_chalmers@4.7676_EC fine.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY(entry=16)

def asym_lorfunc(x, y0, yscale1, f0, gamma1, alpha1, yscale2, f1, gamma2, alpha2):
    lor1 = yscale1 / (1 + ((x - f0) / (gamma1 * (1 + alpha1 * (x - f0))))**2)
    lor2 = yscale2 / (1 + ((x - f1) / (gamma2 * (1 + alpha2 * (x - f1))))**2)
    return y0 + lor1 + lor2

def fit_asym_lor(xdata, ydata, fitparams=None):
    if fitparams is None: 
        fitparams = [None] * 9
    else: 
        fitparams = np.copy(fitparams)

    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 10
    if fitparams[4] is None: fitparams[4] = 0.01
    if fitparams[5] is None: fitparams[5] = max(ydata) - min(ydata)
    if fitparams[6] is None: fitparams[6] = xdata[np.argmax(abs(ydata - fitparams[0])) - 50]  
    if fitparams[7] is None: fitparams[7] = (max(xdata) - min(xdata)) / 10
    if fitparams[8] is None: fitparams[8] = 0.01 


    fitparams = [0 if param is None else param for param in fitparams]

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.curve_fit(asym_lorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        print('Warning: fit failed!')
    return pOpt, pCov


# frequencies = np.linspace(4.0, 5.0, 1000)
# asym_lorentzian_dip_1 = -1.0 / (1 + ((frequencies - 4.8) / (0.01 * (1 + 1 * (frequencies - 4.5))))**2)
# asym_lorentzian_dip_2 = -1.0 / (1 + ((frequencies - 4.2) / (0.05 * (1 + 10 * (frequencies - 4.2))))**2)
# combined_asym_lorentzian_dip = asym_lorentzian_dip_1 + asym_lorentzian_dip_2
# noise = np.random.normal(0, 0.02, frequencies.shape)
# noisy_combined_asym_lorentzian_dip = combined_asym_lorentzian_dip + noise


fitparams = [None, None, 4.6, 0.01, -10, None, 4.17, 0.01, 10]
y = np.abs(y)
pOpt, pCov = fit_asym_lor(x, y, fitparams=None)


plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Noisy Combined Asymmetric Lorentzian Dip', color='black', linestyle='-', alpha=0.7)
plt.plot(x, asym_lorfunc(x, *pOpt), label='Fitted Asymmetric Lorentzian', color='red', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude')

plt.legend()
plt.grid(True)
plt.show()
