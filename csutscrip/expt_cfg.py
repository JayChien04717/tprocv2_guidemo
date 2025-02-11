

tof_config = {
    ## Channel Params. ##
    'gen_ch': 0,
    'ro_ch': 0,
    ## Pulse Params. ##
    'freq': 7100,  # [MHz]
    'pulse_len': 0.2,  # [us]
    'pulse_phase': 0,  # [deg]
    'pulse_gain': 0.8,  # [DAC units]
    ## Readout Params. ##
    'trig_time': 0,  # [us]
    'ro_len': 1.3,  # [us]
    'relax_delay': 0.1,  # [us]
}

config = {
    ## Channel Params. ##
    'res_ch': 0,
    'ro_ch': 0,
    'qubit_ch': 1,

    ## Resonator Pulse Params. ##
    'f_res': 5000,  # [MHz]
    'res_len': 6.0,  # [us]
    'res_phase': 0,  # [deg]
    'res_gain': 0.1,  # [DAC units]

    ## Readout Params. ##
    'trig_time': 0.65,  # [us]
    'ro_len': 8.0,  # [us]
    'relax_delay': 10,  # [us]

    ## Qubit Params. ##
    'f_ge': 4000,  # [MHz]
    'probe_len': 5,
    'qubit_gain': 0.1,
    "pi_gain": 0.2,  # [DAC units]
    "pi2_gain": 0.2 / 2,  # [DAC units]
    'qubit_phase': 0,
    'sigma': 0.05,
    'ramsey_freq': 5,  # [MHz]
}
