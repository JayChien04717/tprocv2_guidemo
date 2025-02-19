# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D


##################
# Define Program #
##################


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
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

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_pulse", t=0)  # play probe pulse
        # self.delay_auto(t=0.05)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
