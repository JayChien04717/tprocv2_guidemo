
hw_cfg = {
    # DAC
    "qubit_ch": [2]*6,  # Qubit Channel Port, Full-speed DAC
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
    "ro_length": 1.0,  # [us]
    "mixer_freq": 5600.0,  # [MHz] - used for mux_ch and interpolated_ch

    # Changes related to the resonator output channel
    "res_freq_ge": [5000.0, 5100.0, 5200.0, 5300.0, 5400.0, 5500.0],  # [MHzx]
    "res_gain_ge": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # [DAC units]
    "res_freq_ef": [5000.0, 5100.0, 5200.0, 5300.0, 5400.0, 5500.0],  # [MHz]
    "res_gain_ef": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # [DAC units]
    "res_length": 1.0,  # [us] (1.0 for res spec)
    # Rotation Angle From QICK Function
    "res_phase": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Threshold for Distinguish g/e, from QICK Function
    "threshold": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    "reps": 10,
    "relax_delay": 10,  # [us]
    "py_avg": 100
}

fit_key = [
    'res_spec_ge',
    'res_punch_out'
    'qubit_spec_ge',
    'time_rabi_ge',
    'power_rabi_ge',
    'Ramsey_ge',
    'SpinEcho_ge',
    'T1_ge',
    'res_spec_ef',
    'qubit_spec_ef',
    'qubit_temp',
    'power_rabi_ef',
    'Ramsey_ef',
    'SingleShot'
]
