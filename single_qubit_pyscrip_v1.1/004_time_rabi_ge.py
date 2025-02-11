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
expt_name = "004_time_rabi_ge"
QubitIndex = 2
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)


##################
# Define Program #
##################

class LengthRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_loop("lenloop", cfg["steps"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse",
                   t=0)  # play probe pulse

        self.delay_auto(t=0.02, tag='waiting')

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


###################
# Experiment sweep parameter
###################
START_LEN = 0.2  # [us]
STOP_LEN = 5  # [us]
STEPS = 100
config.update([('steps', STEPS), ('qubit_length_ge',
              QickSweep1D('lenloop', START_LEN, STOP_LEN))])

###################
# Run the Program
###################

prog = LengthRabiProgram(
    soccfg, reps=10, final_delay=config['relax_delay'], cfg=config)
py_avg = config['py_avg']
iq_list = prog.acquire(soc, soft_avgs=py_avg)
t = prog.get_pulse_param('qubit_pulse', "length", as_array=True)


###################
# Plot
###################
Plot = False

if Plot:
    # plt.plot(freqs,  iq_list[0][0].T[0])
    # plt.plot(freqs,  iq_list[0][0].T[1])
    plt.plot(t, iq_list[0][0].dot([1, 1j]))
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
        "x_name": "x_axis",
        "x_value": t,

        "z_name": "iq_list",
        "z_value": iq_list[0][0].dot([1, 1j])
    }

    result = {
        "T1": "350us",
        "T2": "130us"
    }

    saveh5(file_path, data_dict, result)
