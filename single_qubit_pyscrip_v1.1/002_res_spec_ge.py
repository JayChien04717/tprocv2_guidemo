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
expt_name = "002_res_spec_ge"
QubitIndex = 2
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)


##################
# Define Program #
##################

class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        self.add_loop("freqloop", cfg["steps"])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class Resonator_onetone:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg

    def run(self, reps):
        prog = SingleToneSpectroscopyProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']
        self.iq_list = prog.acquire(soc, soft_avgs=py_avg, progress=True)
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        plt.plot(self.freqs,  np.abs(self.iq_list[0][0].dot([1, 1j])))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("ADC unit (a.u)")
        plt.title("Resonator OneTone Spectroscopy")
        plt.show()

    def save(self):
        data_path = DATA_PATH
        exp_name = expt_name + '_Q' + str(QubitIndex)
        print('Experiment name: ' + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        print('Current data file: ' + file_path)

        data_dict = {
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,

            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }
        saveh5(file_path, data_dict)


###################
# Experiment sweep parameter
###################

START_FREQ = 4000  # [MHz]
STOP_FREQ = 5000  # [MHz]
STEPS = 101
config.update([('steps', STEPS), ('res_freq_ge',
              QickSweep1D('freqloop', START_FREQ, STOP_FREQ))])

###################
# Run the Program
###################

onetone = Resonator_onetone(soccfg, config)
onetone.run(reps=1)
onetone.plot()
onetone.save()
