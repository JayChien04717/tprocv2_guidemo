# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from .async_fun import asyn_run

# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from .module_fitzcu import spectrum_analyze
from .fitting import fitlor, lorfunc
from .plot_utils import plot_final2, plot_final2_plotly
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        if soccfg['gens'][res_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg['res_freq_ge_mixer'], ro_ch=ro_ch)
        else:
            self.declare_gen(ch=res_ch, nqz=2, ro_ch=ro_ch)

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])


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
            name="qubit",
            sigma=cfg["sigma"],
            length=5 * cfg["sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse",
            style="flat_top",
            envelope="qubit",
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

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Qubit_Twotone:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            prog = PulseProbeSpectroscopyProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

    def plot(self):
        f_q = spectrum_analyze(self.freqs, self.iqdata)
        return f_q

    async def liveplot(self, py_avg=1):
        """
        Async liveplot for Qubit ge spectroscopy using PulseProbeSpectroscopyProgram.
        Uses asyn_run(mode="1D") for live plotting, then does Lorentzian fit on final data.
        """
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

        # ===== Interactive FigureWidget =====
        figw = go.FigureWidget(
            data=[
                go.Scatter(
                    x=self.freqs,
                    y=np.zeros_like(self.freqs, dtype=float),
                    mode="lines+markers",
                    name="Amplitude",
                )
            ]
        )
        figw.update_layout(
            title=f"average: 0 / {py_avg}",
            xaxis_title="Frequency (MHz)",
            yaxis_title="ADC units",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        # Run acquisition + live plot
        self.iqdata = await asyn_run(
            prog, self.soc, py_avg, figw,
            title="Qubit ge Spectrum",
            mode="1D"
        )

        # Close interactive widget
        figw.close()

        # ===== Final summary plot with Lorentzian fit =====
        fit_params, error, fig = plot_final2_plotly(
            self.freqs, self.iqdata, x_label="Frequency (MHz)", fitfunc=fitlor, simfunc=lorfunc
        )

        title_text = f"Qubit ge Spectrum, Qubit freq = {fit_params[2]:.6f} MHz"
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor="center"),
            autosize=True,
            margin=dict(l=10, r=10, t=50, b=30, pad=0),
        )
        fig.update_xaxes(constrain="domain")
        fig.update_yaxes(constrain="domain")

        display(fig)

        resonance_freq = float(fit_params[2])  # MHz
        return round(resonance_freq, 6)


    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "003_qubit_spec_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        try:
            self.cfg.pop("qubit_freq_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="TwoTone",
        )
        print(f"Data save to {file_path}")
