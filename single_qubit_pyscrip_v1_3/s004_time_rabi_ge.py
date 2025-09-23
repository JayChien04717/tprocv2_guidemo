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
from tqdm.auto import tqdm
import asyncio
# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator
from .module_fitzcu import lengthrabi_analyze
from .fitting import decaysin, fitdecaysin, fix_phase
from .plot_utils import plot_final
from .yamltool import yml_comment
from .async_fun import asyn_run

##################
# Define Program #
##################


class LengthRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        if soccfg['gens'][res_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg['res_freq_ge_mixer'], ro_ch=ro_ch)
        else:
            self.declare_gen(ch=res_ch, nqz=2, ro_ch=ro_ch)

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
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0.05, tag="waiting")

        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Time_Rabi:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg,  Start_time, Stop_time, Step, liveplot= True):
        self.time_step = np.linspace(Start_time, Stop_time, Step)
        if liveplot:
            self.liveplot(py_avg, Start_time, Stop_time, Step,)
        else:
            iqlst = []
            for i in tqdm(self.time_step, desc="Time sweep"):
                self.cfg["qubit_length_ge"] = i
                prog = LengthRabiProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=False)
                iqlst.append(iq_list[0][0].dot([1, 1j]))
            self.iqlst = np.array(iqlst)

    def plot(self):
        lengthrabi_analyze(self.time_step, self.iqlst)

    async def liveplot(self, py_avg, Start_time, Stop_time, Step):
        """
        Async liveplot for Time Rabi (vary qubit pulse length).
        Uses Plotly FigureWidget for interactive update,
        then fits with decaying sine to extract pi/2 and pi times.
        """
        # Generate time steps
        self.time_step = np.linspace(Start_time, Stop_time, Step)

        # ===== Interactive Plot =====
        figw = go.FigureWidget(
            data=[
                go.Scatter(
                    x=self.time_step,
                    y=np.zeros_like(self.time_step, dtype=float),
                    mode="lines+markers",
                    name="Rabi oscillation",
                )
            ]
        )
        figw.update_layout(
            title=f"average: 0 / {py_avg}",
            xaxis_title="Time (us)",
            yaxis_title="ADC unit",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        display(figw)

        iq_sum = 0
        for avg in tqdm(range(py_avg), desc="average count"):
            # --- Sweep over time steps ---
            iqlst = []
            for t in self.time_step:
                self.cfg["qubit_length_ge"] = t
                prog = LengthRabiProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
                iqlst.append(iq_list[0][0].dot([1, 1j]))

            iqlst = np.array(iqlst)

            # Running average
            iq_sum = iqlst if avg == 0 else iq_sum + iqlst
            self.iqdata = iq_sum / (avg + 1)

            # Update plot
            with figw.batch_update():
                figw.data[0].y = np.abs(self.iqdata)
                figw.layout.title.text = f"average: {avg + 1} / {py_avg}"

            await asyncio.sleep(0)

        # Close interactive widget
        figw.close()

        # ===== Final fit plot =====
        fit_params, error, fig = plot_final(
            self.time_step, self.iqdata, "Time (us)", fitdecaysin, decaysin
        )
        fig.suptitle("Time Rabi ge")
        fig.tight_layout()

        # Extract pi and pi/2 pulse lengths
        pi_gain, pi2_gain = fix_phase(fit_params)
        return round(pi_gain, 6), round(pi2_gain, 6)


    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "004_time_rabi_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)

        dict_val = yml_comment(self.cfg)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.time_step * 1e-6},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqlst},
            comment=(f"{dict_val}"),
            tag="Rabi",
        )

        print(f"Data save to {file_path}")
