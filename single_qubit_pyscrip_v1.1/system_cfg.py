from qick import *
from qick.pyro import make_proxy
from qick import QickConfig
import Pyro4
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
soc, soccfg = make_proxy(ns_host="192.168.20.46",
                         ns_port=8888, proxy_name="myqick")
print(soccfg)

# Where do you want to save data
DATA_PATH = r"C:\Users\SQC\Desktop\QICK\jay scrip\tprocv2\data"

hw_cfg = {
    # DAC
    "qubit_ch": [1]*6,  # Qubit Channel Port, Full-speed DAC
    "res_ch": [0]*6,  # Single Tone Readout Port, Full-speed DAC
    "qubit_ch_ef": [1]*6,  # Qubit ef Channel, Full-speed DAC
    "mux_ch": 12,
    "nqz_qubit": 2,
    "nqz_res": 2,
    # ADC
    "ro_ch": [0] * 6,  # tproc configured readout channel
    "mux_ro_chs": [2, 3, 4, 5, 6, 7]
}


# Readout Configuration
readout_cfg = {
    "trig_time": 0.50,  # [Clock ticks] - get this value from TOF experiment
    "ro_length": 1.5,  # [us]
    "mixer_freq": 5600,  # [MHz] - used for mux_ch and interpolated_ch

    # Changes related to the resonator output channel
    "res_freq_ge": [5000, 5100, 5200, 5300, 5400, 5500],  # [MHzx]
    "res_gain_ge": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # [DAC units]
    "res_freq_ef": [5000, 5100, 5200, 5300, 5400, 5500],  # [MHz]
    "res_gain_ef": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # [DAC units]
    "res_length": 1.0,  # [us] (1.0 for res spec)
    "res_phase": [0, 0, 0, 0, 0, 0],  # Rotation Angle From QICK Function
    # Threshold for Distinguish g/e, from QICK Function
    "threshold": [0, 0, 0, 0, 0, 0],
}

# Qubit Configuration
qubit_cfg = {
    # Freqs of Qubit g/e Transition
    "qubit_freq_ge": [4000, 4000, 4000, 4000, 4000, 4000],
    "qubit_gain_ge": [0.1, 0.1, 0.1, 0.1, 0.1],
    "qubit_length_ge": 5,  # [us] for Constant Pulse
    # [MHz] Freqs of Qubit e/f Transition
    "qubit_freq_ef": [4000, 4000, 4000, 4000, 4000, 4000],
    # [0.01, 0.05, 0.05, 0.05, 0.01, 0.5], # [DAC units] Pulse Gain
    "qubit_gain_ef": [0.0891, 0.086, 0.03, 0.03, 0.03, 0.1],
    "qubit_length_ef": 25.0,  # [us] for Constant Pulse
    "qubit_phase": 0,  # [deg]
    # [us] for Gaussian Pulse
    "sigma": [0.1/5, 0.1/5, 0.1/5, 0.1/5, 0.1/5, .1/5],
    # [us] for Gaussian Pulse
    "sigma_ef": [0.1/5, 0.1/5, 0.1/5, 0.1/5, 0.1/5, .1/5],
    "ramsey_freq": 2  # [MHz]
}


expt_cfg = {
    "reps": 1,
    "soft_avgs": 100,
    "relax_delay": 10,  # [us]
    "py_avg": 100
}
