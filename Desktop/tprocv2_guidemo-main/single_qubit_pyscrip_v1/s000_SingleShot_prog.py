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
from singleshotplot import hist
from system_cfg import *
from system_tool import select_config_idx, saveshot, get_next_filename
from pprint import pprint
# ----- Experiment configurations ----- #
expt_name = "000_SingleShot_gef"
QubitIndex = 0
Qubit = 'Q' + str(QubitIndex)
config = select_config_idx(
    hw_cfg, readout_cfg, qubit_cfg, expt_cfg, idx=QubitIndex)


##################
# Define Program #
##################

# Separate g and e per each experiment defined.

class SingleShotProgram_g(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["shots"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.delay_auto(0.01, tag='wait')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleShotProgram_e(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["shots"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(0.01, tag='wait')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleShotProgram_f(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["shots"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp_ge",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_ge_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ge",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp_ef",
                       sigma=cfg['sigma'], length=cfg['sigma_ef']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_ef_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ef",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ef'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"],  name="qubit_ge_pulse", t=0)
        self.delay_auto(0.01, tag='wait1')
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_ef_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qubit_ch"],  name="qubit_ge_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleShot_gef:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg

    def run(self, shot_f=False):
        shot_g = SingleShotProgram_g(
            soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        shot_e = SingleShotProgram_e(
            soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)

        iq_list_g = shot_g.acquire(soc, soft_avgs=1, progress=True)
        iq_list_e = shot_e.acquire(soc, soft_avgs=1, progress=True)

        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
        if shot_f:
            shot_f = SingleShotProgram_f(
                soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
            iq_list_f = shot_f.acquire(soc, soft_avgs=1, progress=True)
            I_f = iq_list_f[0][0].T[0]
            Q_f = iq_list_f[0][0].T[1]

        if shot_f:
            self.data = {'Ig': I_g, 'Qg': Q_g,
                         'Ie': I_e, 'Qe': Q_e,
                         'If': I_f, 'Qf': Q_f, }
        else:
            self.data = {'Ig': I_g, 'Qg': Q_g,
                         'Ie': I_e, 'Qe': Q_e}

    def plot(self, fid_avg=False, fit=False, normalize=False):
        hist(
            self.data,
            amplitude_mode=False,
            ps_threshold=None,
            theta=None,
            plot=True,
            verbose=True,
            fid_avg=False,
            fit=True,
            fitparams=[None, None, 20, None, None, 20],
            normalize=False,
            title=None,
            export=False,
        )

    def save(self, result: dict = None):
        data_path = DATA_PATH
        exp_name = expt_name + '_Q' + str(QubitIndex)
        print('Experiment name: ' + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        print('Current data file: ' + file_path)

        data_dict = self.data
        if result is not None:
            saveshot(file_path, data_dict, result)
        else:
            saveshot(file_path, data_dict)


if __name__ == "__main__":
    ###################
    # Experiment sweep parameter
    ###################

    Shots = 5000
    config.update([('shots', Shots)])

    ###################
    # Run the Program
    ###################

    ss = SingleShot_gef(soccfg, config)
    ss.run(shot_f=False)
    ss.plot()
    ss.save()
