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
expt_name = "001_tof"
QubitIndex = 1
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)

pprint(config)

##################
# Define Program #
##################


class LoopbackProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['freq'], gen_ch=gen_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="myconst", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['res_ch'], name="myconst", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0)


###################
# Run the Program
###################

prog = LoopbackProgram(
    soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
iq_list = prog.acquire_decimated(soc, soft_avgs=config['soft_avgs'])
t = prog.get_time_axis(ro_index=0)


###################
# Plot
###################
Plot = True

if Plot:
    plt.plot(t,  iq_list[0].T[0])
    plt.plot(t,  iq_list[0].T[1])
    plt.plot(t, np.abs(iq_list[0].dot([1, 1j])))
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
        "z_value": iq_list[0].dot([1, 1j])
    }

    result = {
        "T1": "350us",
        "T2": "130us"
    }

    saveh5(file_path, data_dict, config, result)
# %%
