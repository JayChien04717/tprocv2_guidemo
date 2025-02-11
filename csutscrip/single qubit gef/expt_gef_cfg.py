

config = {
    ## Channel Params. ##
    'res_ch': 0,
    'ro_ch': 0,
    'qubit_ch': 1,

    ## Resonator Pulse Params. ##
    'res_freq_ge': 5000,  # [MHz]
    'nqz_res': 2,
    'res_length': 6.0,  # [us]
    'res_phase': 0,  # [deg]
    'res_gain_ge': 0.1,  # [DAC units]

    ## Readout Params. ##
    'trig_time': 0.65,  # [us]
    'ro_length': 6.5,  # [us]
    'relax_delay': 10,  # [us]

    ## Qubit Params. ##
    'nqz_qubit': 2,
    'qubit_freq_ge': 4000,  # [MHz]
    'pi_gain_ge': 0.1,
    'pi2_gain_ge': 0.1 / 2,

    'qubit_freq_ef': 3500,  # [MHz]
    'pi_gain_ef': 0.1,
    'pi2_gain_ef': 0.1 / 2,

    'qubit_phase': 0,
    'probe_len': 5,
    'sigma': 0.01,
    'ramsey_freq': 5,  # [MHz]
}
