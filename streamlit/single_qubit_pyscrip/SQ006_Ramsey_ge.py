# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2

from qick.asm_v2 import QickSpan, QickSweep1D


##################
# Define Program #
##################


class RamseyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qubit_mixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("waitloop", cfg["steps"])

        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="const",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse1",
            ro_ch=ro_ch,
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi2_gain_ge"],
        )

        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse2",
            ro_ch=ro_ch,
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            # current phase + time * 2pi * ramsey freq
            phase=cfg["qubit_phase"] + cfg["wait_time"] * 360 * cfg["ramsey_freq"],
            gain=cfg["qubit_pi2_gain_ge"],
        )

        if cfg["apply_cool"]:
            from .qick_pulse_setup import declare_cooling_pulse

            declare_cooling_pulse(self, cfg)

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg["apply_cool"]:
            self.pulse(ch=cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(cfg["cool_delay"])
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
