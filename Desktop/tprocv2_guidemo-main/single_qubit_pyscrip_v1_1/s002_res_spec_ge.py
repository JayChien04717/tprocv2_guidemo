# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D
# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import  get_next_filename_labber, hdf5_generator
from .module_fitzcu import resonator_circlefit, resonator_analyze



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
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg):
        prog = SingleToneSpectroscopyProgram(
            self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)

        self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        param = resonator_analyze(self.freqs,  self.iq_list[0][0].dot([1, 1j]))
        return param
    
    def plot_circle(self):
        param = resonator_circlefit(self.freqs,  self.iq_list[0][0].dot([1, 1j]))
        return param

    def saveLabber(self, qb_idx):
        expt_name = "s002_onetone" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        hdf5_generator(
                filepath=file_path,
                x_info={'name': 'Frequency', 'unit': "Hz",
                        'values': self.freqs*1e6},

                z_info={'name': 'Signal', 'unit': 'ADC unit',
                        'values':  self.iq_list[0][0].dot([1, 1j])},
                comment=(),
                tag= 'OneTone'
        )
        print(f'Data save to {file_path}')


if __name__ =='__main__':
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
