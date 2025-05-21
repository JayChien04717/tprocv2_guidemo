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
expt_name = "010_qubit_spec_ef"
QubitIndex = 0
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)

##################
# Define Program #
##################


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        qubit_ch_ef = cfg['qubit_ch_ef']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        if soccfg['gens'][qubit_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qmixer_freq'])
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        if soccfg['gens'][qubit_ch_ef]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=qubit_ch_ef, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qmixer_freq'])
        else:
            self.declare_gen(ch=qubit_ch_ef, nqz=cfg['nqz_qubit'])

        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        self.add_loop("freqloop", cfg["steps"])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ef'], gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ef'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ef'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pi_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_pi_gain_ge'],
                       )

        self.add_pulse(ch=qubit_ch_ef, name="qubit_pulse_ef", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ef'],
                       freq=cfg['qubit_freq_ef'],
                       phase=0,
                       gain=cfg['qubit_gain_ef'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qubit_ch_ef"],
                   name="qubit_pulse_ef", t=0)  # play probe pulse
        self.delay_auto(0.01)
        if self.cfg['ge_ref']:
            self.delay_auto(0.01)
            self.pulse(ch=cfg['qubit_ch'], name="qubit_pi_pulse", t=0)
        self.delay_auto(0.05)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


###################
# Run the Program
###################

START_FREQ = 4400  # [MHz]
STOP_FREQ = 4550  # [MHz]
STEPS = 101
config.update([('steps', STEPS),
               ('qubit_freq_ef', QickSweep1D('freqloop', START_FREQ, STOP_FREQ)),
               ("ge_ref", True)
              ])


###################
# Run the Program
###################

qspec = PulseProbeSpectroscopyProgram(
    soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
py_avg = config['py_avg']
iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
freqs = qspec.get_pulse_param('qubit_pulse_ef', "freq", as_array=True)


###################
# Plot
###################

Plot = True

if Plot:
    # plt.plot(freqs,  iq_list[0][0].T[0])
    # plt.plot(freqs,  iq_list[0][0].T[1])
    plt.plot(freqs, np.abs(iq_list[0][0].dot([1, 1j])))
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
        "x_value": freqs,

        "z_name": "iq_list",
        "z_value": iq_list[0][0].dot([1, 1j])
    }

    result = {
        "T1": "350us",
        "T2": "130us"
    }

    saveh5(file_path, data_dict, result)
# %%
