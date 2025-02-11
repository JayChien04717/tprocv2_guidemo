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
expt_name = "002b_res_punchout_ge"
QubitIndex = 2
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)


##################
# Define Program #
##################

class SingleToneSpectroscopyPunchoutProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])
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


class SingleToneSpectroscopyPunchout:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg

    def run(self, reps):
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = config['py_avg']
        self.iq_list = prog.acquire(soc, soft_avgs=py_avg, progress=True)
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def plot(self):
        avg_abs, avg_angle = (np.abs(self.iq_list[0][0].dot([1, 1j])),
                              np.angle(self.iq_list[0][0].dot([1, 1j])))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, d in enumerate([avg_abs, avg_angle]):
            if i == 0:
                pcm = axes[i].pcolormesh(
                    self.freqs, self.gains, d, shading="Auto")
            else:
                pcm = axes[i].pcolormesh(
                    self.freqs, self.gains, np.unwrap(d), shading="Auto", cmap="bwr")
            axes[i].set_ylabel("Gain")
            axes[i].set_xlabel("Freq(MHz)")
            axes[i].set_title("Amp" if i == 0 else "IQ phase (rad)")
            plt.colorbar(pcm, ax=axes[i])
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
            "y_name": "DAC Gain (a.u)",
            "y_value": self.gains,
            "z_name": "iq_list",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }
        saveh5(file_path, data_dict)
###################
# Experiment sweep parameter
###################


START_FREQ = 5000  # [MHz]
STOP_FREQ = 6000  # [MHz]
STEPS_freq = 100

START_gain = 0.1  # [MHz]
STOP_gain = 0.5  # [MHz]
STEPS_gain = 5
config.update([('f_steps', STEPS_freq), ('res_freq_ge', QickSweep1D('freqloop', START_FREQ, STOP_FREQ)),
               ('g_steps', STEPS_freq), ('res_gain_ge', QickSweep1D('gainloop', START_gain, STOP_gain))])

###################
# Run the Program
###################
punchout = SingleToneSpectroscopyPunchout(soccfg, config)
punchout.run(reps=1)
punchout.plot()
