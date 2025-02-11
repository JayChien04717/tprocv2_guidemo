import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import sys, os
os.chdir(r'C:\Users\QEL\Desktop\fit_pack\data')
sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
import Labber
np.bool= bool
np.float = np.float64
res = r'C:\Users\QEL\Desktop\fit_pack\data\twotone\Normalized_Self_Lamb[2]_R_@671_uA_two_tone_012.hdf5'
fr = Labber.LogFile(res)
(x, y) = fr.getTraceXY(entry=6)



def threelorfunc(x, *p):
    y0, yscale1, x0_1, xscale1, yscale2, x0_2, xscale2, yscale3, x0_3, xscale3 = p
    return (
        y0 + 
        yscale1 / (1 + (x - x0_1)**2 / xscale1**2) + 
        yscale2 / (1 + (x - x0_2)**2 / xscale2**2) + 
        yscale3 / (1 + (x - x0_3)**2 / xscale3**2)
    )


def fitthreelor(xdata, ydata, fitparams=None):
    if fitparams is None: 
        fitparams = [None] * 10
    else: 
        fitparams = np.copy(fitparams)

    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1]) / 2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata) - min(xdata)) / 3
    if fitparams[4] is None: fitparams[4] = max(ydata) - min(ydata)
    if fitparams[5] is None: fitparams[5] = xdata[np.argmax(abs(ydata - fitparams[0]))]  
    if fitparams[6] is None: fitparams[6] = (max(xdata) - min(xdata)) / 5
    if fitparams[7] is None: fitparams[7] = max(ydata) - min(ydata)
    if fitparams[8] is None: fitparams[8] = xdata[np.argmax(abs(ydata - fitparams[0]))] 
    if fitparams[9] is None: fitparams[9] = (max(xdata) - min(xdata)) / 5

    fitparams = [0 if param is None else param for param in fitparams]
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.curve_fit(threelorfunc, xdata, ydata, p0=fitparams)
    except RuntimeError: 
        print('Warning: fit failed!')
    return pOpt, pCov

frequencies = np.linspace(4.0, 5.0, 1000)
lorentzian_dip_1 = -1.0 / (1 + (frequencies - 4.5)**2 / 0.005**2)
lorentzian_dip_2 = -0.6 / (1 + (frequencies - 4.2)**2 / 0.005**2)
lorentzian_dip_3 = -0.4 / (1 + (frequencies - 4.35)**2 / 0.005**2)
combined_lorentzian_dip = lorentzian_dip_1 + lorentzian_dip_2 + lorentzian_dip_3
noise = np.random.normal(0, 0.02, frequencies.shape)
noisy_combined_lorentzian_dip = combined_lorentzian_dip + noise


fitparams = [None, None, 4.5, 0.05, None, 4.2, 0.05, None, 4.35, 0.05]
pOpt, pCov = fitthreelor(frequencies, noisy_combined_lorentzian_dip, fitparams)


plt.figure(figsize=(10, 6))
plt.plot(frequencies, noisy_combined_lorentzian_dip, label='Noisy Combined Lorentzian Dip', color='black', linestyle='-', alpha=0.7)
plt.plot(frequencies, threelorfunc(frequencies, *pOpt), label='Fitted Lorentzian', color='red', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude')
plt.title('Noisy Lorentzian Dips at 4.5 GHz, 4.2 GHz, and 4.35 GHz with Fitting')
plt.legend()
plt.grid(True)
plt.show()


