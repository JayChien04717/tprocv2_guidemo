# %%
# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
# for now, all the tProc v2 classes need to be individually imported (can't use qick.*)
# the main program class
from qick.asm_v2 import AveragerProgramV2
# for defining sweeps
from qick.asm_v2 import QickSpan, QickSweep1D
# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
import datetime
from system_cfg import *
from system_tool import select_config_idx, saveh5, get_next_filename
from pprint import pprint
# ----- Experiment configurations ----- #
expt_name = "012_Ramsey_ef"
QubitIndex = 2
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)


##################
# Define Program #
##################

class RamseyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        qubit_ch_ef = cfg['qubit_ch_ef']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.declare_gen(ch=qubit_ch_ef, nqz=cfg['nqz_qubit'])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_loop("waitloop", cfg["steps"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pi_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch_ef, name="ramp1",
                       sigma=cfg['sigma_ef'], length=cfg['sigma_ef']*5, even_length=True)
        self.add_pulse(ch=qubit_ch_ef, name="qubit_pulse1", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp1",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'] / 2,
                       )

        self.add_pulse(ch=qubit_ch_ef, name="qubit_pulse2", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp1",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'] +
                       cfg['wait_time']*360*cfg['ramsey_freq'],
                       gain=cfg['qubit_gain_ef'] / 2,
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qubit_ch_ef"], name="qubit_pulse1", t=0)
        self.delay_auto(cfg['wait_time']+0.01, tag='wait')
        self.pulse(ch=self.cfg["qubit_ch_ef"], name="qubit_pulse2", t=0)
        self.delay_auto(0.01)
        # Return back to ge to get highest SNR. If don't want can comment out
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.01)

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


###################
# Experiment sweep parameter
###################

START_TIME = 0.0  # [us]
STOP_TIME = 100  # [us]
STEPS = 100
config.update([('steps', STEPS), ('wait_time',
              QickSweep1D('waitloop', START_TIME, STOP_TIME))])

###################
# Run the Program
###################

ramsey = RamseyProgram(
    soccfg, reps=100, final_delay=config['relax_delay'], cfg=config)
py_avg = 10
iq_list = ramsey.acquire(soc, soft_avgs=py_avg, progress=True)
delay_times = ramsey.get_time_param('wait', "t", as_array=True)


###################
# Plot
###################

Plot = True

if Plot:
    # plt.plot(freqs,  iq_list[0][0].T[0])
    # plt.plot(freqs,  iq_list[0][0].T[1])
    plt.plot(delay_times, iq_list[0][0].dot([1, 1j]))
    plt.show()

#####################################
# ----- Saves data to a file ----- #
#####################################

Save = True
if Save:
    data_path = "./data"
    labber_data = "./data/Labber"
    exp_name = expt_name + '_Q' + str(QubitIndex)
    print('Experiment name: ' + exp_name)
    file_path = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + file_path)

    data_dict = {
        "x_name": "Ramsey time(us)",
        "x_value": delay_times,

        "z_name": "iq_list",
        "z_value": iq_list[0][0].dot([1, 1j])
    }

    result = {
        "T1": "350us",
        "T2": "130us"
    }

    saveh5(file_path, data_dict, result)
