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
from .system_tool import get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from .module_fitzcu import spectrum_analyze, post_rotate
from .fitting import *
from .yamltool import yml_comment
from IPython.display import display, clear_output

### AllXY Sequence ###
sequence = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]


class AllXYprogram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        # self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

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
        # if cfg["qb_pulse_style"] == "arb":
        self.add_pulse(
            ch=qubit_ch,
            name="x180",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=0,
            gain=cfg["qubit_pi_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="x90",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=0,
            gain=cfg["qubit_pi2_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="y180",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=90,
            gain=cfg["qubit_pi_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="y90",
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=90,
            gain=cfg["qubit_pi2_gain_ge"],
        )
        # elif cfg["qb_pulse_style"] == "flat_top":
        #     self.add_pulse(
        #         ch=qubit_ch,
        #         name="x180",
        #         style="flat_top",
        #         envelope="ramp",
        #         freq=cfg["qubit_freq_ge"],
        #         phase=0,
        #         gain=cfg["qubit_pi_gain_ge"],
        #         length=cfg["flat_top_len"],
        #     )
        #     self.add_pulse(
        #         ch=qubit_ch,
        #         name="x90",
        #         style="flat_top",
        #         envelope="ramp",
        #         freq=cfg["qubit_freq_ge"],
        #         phase=0,
        #         gain=cfg["qubit_pi2_gain_ge"],
        #         length=cfg["flat_top_len"],
        #     )
        #     self.add_pulse(
        #         ch=qubit_ch,
        #         name="y180",
        #         style="flat_top",
        #         envelope="ramp",
        #         freq=cfg["qubit_freq_ge"],
        #         phase=90,
        #         gain=cfg["qubit_pi_gain_ge"],
        #         length=cfg["flat_top_len"],
        #     )
        #     self.add_pulse(
        #         ch=qubit_ch,
        #         name="y90",
        #         style="flat_top",
        #         envelope="ramp",
        #         freq=cfg["qubit_freq_ge"],
        #         phase=90,
        #         gain=cfg["qubit_pi2_gain_ge"],
        #         length=cfg["flat_top_len"],
        #     )

    def apply_cool(self, cfg):
        cool_ch1 = cfg["cool_ch1"]
        cool_ch2 = cfg["cool_ch2"]
        if self.soccfg["gens"][cool_ch1]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch1, nqz=cfg["nqz_cool_ch1"], mixer_freq=cfg["cool_mixer1"]
            )
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg["nqz_cool_ch1"])

        if self.soccfg["gens"][cool_ch2]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=cool_ch2, nqz=cfg["nqz_cool_ch2"], mixer_freq=cfg["cool_mixer2"]
            )
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg["nqz_cool_ch2"])

        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="const",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_2"],
            phase=0,
            gain=cfg["cool_gain_2"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg["cooling"] is True:
            self.apply_cool(cfg)
            self.pulse(ch=self.cfg["cool_ch1"], name="cool_pulse1", t=0)
            self.pulse(ch=self.cfg["cool_ch2"], name="cool_pulse2", t=0)
            self.delay_auto(0.5, tag="Ring down")
        else:
            pass

        ## allxy gates ##
        if cfg["allxy_gates"][0] == "I":
            pass
        else:
            self.pulse(
                ch=self.cfg["qubit_ch"], name=f"{self.cfg['allxy_gates'][0]}", t=0
            )
        self.delay_auto(0.01)
        if cfg["allxy_gates"][1] == "I":
            pass
        else:
            self.pulse(
                ch=self.cfg["qubit_ch"], name=f"{self.cfg['allxy_gates'][1]}", t=0
            )

        ## readout pulse ##
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class AllXY:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            print("This program is not supported in liveplot mode")
        else:
            allxy_lst = []
            for gate in tqdm(sequence):
                self.cfg["allxy_gates"] = gate
                allxy = AllXYprogram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = allxy.acquire(self.soc, rounds=py_avg, progress=False)

                allxy_lst.append(iq_list[0][0].dot([1, 1j]))
            self.allxy_lst = np.array(allxy_lst)

    def plot(self):
        amp = np.abs(self.allxy_lst)
        plt.figure(figsize=(10, 5))
        plt.plot(amp, "bo", label="Data")
        plt.plot(
            [np.min(amp)] * 5 + [np.mean(amp)] * 12 + [np.max(amp)] * 4,
            "r-",
            label="Reference Line",
        )
        plt.xticks(np.arange(len(sequence)), sequence, rotation=45)
        plt.ylabel(r"$F_{\left|1\right\rangle}$")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "003_qubit_spec_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("qubit_freq_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)
        if save_sim:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
                y_info={"name": "simulate", "unit": "None", "values": np.array([0, 1])},
                z_info={
                    "name": "Signal",
                    "unit": "ADC unit",
                    "values": np.array([self.iqdata, self.sim]),
                },
                comment=(f"{dict_val}"),
                tag="TwoTone",
            )
        else:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
                z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
                comment=(f"{dict_val}"),
                tag="TwoTone",
            )
        print(f"Data save to {file_path}")
