# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from  .async_fun import asyn_run
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

        if soccfg['gens'][res_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg['res_freq_ge_mixer'], ro_ch=ro_ch)
        else:
            self.declare_gen(ch=res_ch, nqz=2, ro_ch=ro_ch)
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
            return self.liveplot(py_avg)
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

    async def liveplot(self, py_avg=1):
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

        z0 = np.zeros((len(self.gains), len(self.freqs)), dtype=float)
        figw = go.FigureWidget(
            data=[
                go.Heatmap(
                    x=self.freqs,
                    y=self.gains,
                    z=z0,
                    zmin=0.0,
                    zmax=1.0,
                    showscale=True,
                )
            ]
        )
        figw.update_layout(
            title="Resonator ge Punchout",
            xaxis_title="Frequency (MHz)",
            yaxis_title="DAC Gain",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        self.iqdata, z_final = await asyn_run(prog, self.soc, py_avg, figw, title="Resonator ge Punchout", mode="2D")
        figw.close() 

        fig_final = go.Figure(
            data=[
                go.Heatmap(
                    x=self.freqs,
                    y=self.gains,
                    z=z_final,
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(
                        title="Normalized Amplitude",
                        thickness=12,
                        x=1.0,
                        xpad=4,
                        len=0.9,
                    ),
                )
            ]
        )
        fig_final.update_layout(
            title="Resonator ge Punchout",
            xaxis_title="Frequency (MHz)",
            yaxis_title="DAC Gain",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        fig_final.update_xaxes(constrain="domain")
        fig_final.update_yaxes(constrain="domain")
        display(fig_final)





    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "002b_res_ge_punchout" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
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
