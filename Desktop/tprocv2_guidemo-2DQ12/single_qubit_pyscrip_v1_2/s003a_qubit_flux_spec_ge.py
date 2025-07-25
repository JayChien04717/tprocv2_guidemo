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
from .fitting import *
from .yamltool import yml_comment
from IPython.display import display, clear_output
from .YOKOGS200 import YOKOGS200
import pyvisa
##################
# Define Program #
##################


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
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

        self.add_loop("freqloop", cfg["steps"])
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
        self.add_gauss(
            ch=qubit_ch,
            name="quibit",
            sigma=cfg["sigma"],
            length=5 * cfg["sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse",
            style="flat_top",
            envelope="quibit",
            length=cfg["qubit_length_ge"],
            freq=cfg["qubit_freq_ge"],
            phase=0,
            gain=cfg["qubit_gain_ge"],
        )

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
        self.add_gauss(
            ch=cool_ch1,
            name="cooling1",
            sigma=0.004,
            length=0.004 * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            envelope="cooling1",
            style="flat_top",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=0.004,
            length=0.004 * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            envelope="cooling2",
            style="flat_top",
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

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Qubit_Twotone_Flux:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            pass

    def liveplot_yoko(
        self,
        py_avg,
        yoko_value: np.ndarray,
        yoko_inst: str = None,
        mode: str = "current",
    ):
        rm = pyvisa.ResourceManager()
        yoko = YOKOGS200(yoko_inst, rm)

        fig, ax = plt.subplots(figsize=(6, 4))
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
        self.iqdata = np.zeros((len(yoko_value), len(self.freqs)))
        self.yoko_currnet = yoko_value

        mesh = ax.pcolormesh(
            yoko_value * 1e3,
            self.freqs,
            self.iqdata.T,
            shading="nearest",
            cmap="viridis",
        )
        cbar = fig.colorbar(mesh, ax=ax, label="Signal (a.u.)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_xlabel("Current (mA)" if mode == "current" else "Voltage (mV)")

        for idx, curr in tqdm(enumerate(yoko_value)):
            if mode == "current":
                yoko.SetMode("current")
                yoko.SetCurrent(curr)
                title_str = f"Qubit Twotone Current = {curr * 1e3:.3f} mA : {idx + 1}/{len(yoko_value)}"
            elif mode == "voltage":
                yoko.SetMode("voltage")
                yoko.SetVoltage(curr)
                title_str = f"Qubit Twotone Voltage = {curr * 1e3:.3f} mV : {idx + 1}/{len(yoko_value)}"

            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
            iq = iq_list[0][0].dot([1, 1j])
            iq_abs = np.abs(iq)

            self.iqdata[idx, :] = iq_abs
            mesh.set_array(self.iqdata.T.ravel())
            ax.set_title(title_str)
            vmin = np.nanmin(self.iqdata)
            vmax = np.nanmax(self.iqdata)
            mesh.set_clim(vmin, vmax)
            cbar.update_normal(mesh)
            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        plt.close(fig)

        ### Final plot ###
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(
            yoko_value * 1e3,
            self.freqs,
            self.iqdata.T,
            shading="nearest",
            cmap="viridis",
        )
        ax.set_title("Qubit Twotone Flux")
        ax.set_xlabel("Current (mA)" if mode == "current" else "Voltage (mV)")
        ax.set_ylabel("Frequency (MHz)")

    def saveLabber(self, qb_idx):
        expt_name = "003_qubit_flux_spec_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        try:
            self.cfg.pop("qubit_freq_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            y_info={"name": "Yoko", "unit": "A", "values": self.yoko_currnet},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="TwoTone",
        )
        print(f"Data save to {file_path}")
