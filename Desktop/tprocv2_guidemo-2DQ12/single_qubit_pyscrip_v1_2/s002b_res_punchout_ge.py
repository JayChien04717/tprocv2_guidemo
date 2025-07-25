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
from .module_fitzcu import post_rotate
from .fitting import *
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class SingleToneSpectroscopyPunchoutProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleToneSpectroscopyPunchout:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = SingleToneSpectroscopyPunchoutProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
            self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def plot(self):
        data = np.abs(post_rotate(self.iqdata))  # shape: (n_gain, n_freq)
        data_norm = np.array(
            [
                (row - np.min(row)) / (np.max(row) - np.min(row))
                if np.max(row) != np.min(row)
                else row
                for row in data
            ]
        )
        pcm = plt.pcolormesh(self.freqs, self.gains, data_norm)
        plt.title("Resonator Punch Out")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Dac Gains [a.us]")
        plt.colorbar(pcm)

    def liveplot(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            data = np.abs(post_rotate(self.iqdata))  # shape: (n_gain, n_freq)
            data_norm = np.array(
                [
                    (row - np.min(row)) / (np.max(row) - np.min(row))
                    if np.max(row) != np.min(row)
                    else row
                    for row in data
                ]
            )
            ax.cla()
            im = ax.pcolorfast(self.freqs, self.gains, data_norm)
            ax.pcolorfast(self.freqs, self.gains, data_norm)
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Dac Gain")

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)

        ax.set_title(f"Resonator ge Punchout")
        ax.pcolorfast(self.freqs, self.gains, data_norm)
        fig.colorbar(im, ax=ax, label="Normalized Amplitude")

    def saveLabber(self, qb_idx, yoko_current=None):
        expt_name = "002b_res_ge_punchout" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("res_freq_ge")
            self.cfg.pop("res_gain_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            y_info={"name": "DAC Gains", "unit": "a.u.", "values": self.gains},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="OneTone",
        )
        print(f"Data save to {file_path}")
