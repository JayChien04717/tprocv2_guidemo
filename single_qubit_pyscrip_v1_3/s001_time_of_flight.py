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
from .system_tool import hdf5_generator, get_next_filename_labber


##################
# Define Program #
##################


class LoopbackProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        if self.soccfg["gens"][res_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg["res_freq_ge"], ro_ch=ro_ch)
        else:
            self.declare_gen(ch=res_ch, nqz=2, ro_ch=ro_ch)

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch)
        self.add_gauss(
            ch=res_ch,
            name="TOF",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="loopback_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="TOF",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="loopback_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0,  ddr4=True)

class TOF:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg=1, liveplot=True):
        if liveplot:
            return self.liveplot(py_avg)
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        self.iq_list = prog.acquire_decimated(self.soc, rounds=py_avg)
        self.t = prog.get_time_axis(ro_index=0)

    def plot(self, threshold=1.5):
        if self.iq_list is not None:
            plt.plot(self.t, self.iq_list[0].T[0])
            # plt.plot(self.t, self.iq_list[0].T[1])
            # plt.plot(self.t, np.abs((self.iq_list[0]).dot([1, 1j])))
            plt.xlabel("Time (us)")
            plt.ylabel("a.u")
            # plt.title("Time Of Flight")

            mean = np.mean(np.abs(self.iq_list[0].dot([1, 1j])))
            plt.axvline(
                self.t[
                    np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > threshold * mean)
                ],
                c="r",
                ls="--",
            )
            plt.title(
                f"Time of Flight Experiment, trig = {round(self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > 1.5 * mean)], 2)} us"
            )

        else:
            print("No data to plot. Run the experiment first.")

    async def liveplot(self, py_avg=1):
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        self.t = prog.get_time_axis(ro_index=0)


        figw = go.FigureWidget(
            data=[go.Scatter(x=self.t, y=np.zeros_like(self.t), mode="lines+markers", name="|IQ|")]
        )
        figw.update_layout(
            title=f"average: 0 / {py_avg}",
            xaxis_title="Time (us)",
            yaxis_title="ADC unit"
        )
        display(figw)

        self.selected_x = None
        def handle_click(trace, points, selector):
            if points.xs:  
                xval = points.xs[0]
                self.selected_x = xval
                print(f"👉 Selected x = {xval:.4f}")

                figw.data = tuple([d for d in figw.data if d.name != "axvline"])
                figw.add_trace(
                    go.Scatter(
                        x=[xval, xval],
                        y=[0, max(figw.data[0].y) if len(figw.data[0].y) else 1],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="axvline"
                    )
                )
        figw.data[0].on_click(handle_click)

        iq_list = await asyn_run(prog, self.soc, py_avg, figw, title="TOF", mode="decimate")
        self.iq_list = iq_list
        return iq_list


    def saveLabber(self, qb_idx):
        expt_name = "s001_tof" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.t * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "ADC unit",
                "values": self.iq_list,
            },
            comment=(),
            tag="OneTone",
        )
        print(f"Data save to {file_path}")
