# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm
from .async_fun import asyn_run
# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator, auto_unit
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class SingleToneSpectroscopyProgram_yoko(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        if soccfg['gens'][res_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=res_ch, nqz=2, mixer_freq=cfg['res_freq_ge_mixer'], ro_ch=ro_ch)
        else:
            self.declare_gen(ch=res_ch, nqz=2, ro_ch=ro_ch)
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

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class SingleToneSpectroscopyProgram_hardware(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        flux_ch = cfg["flux_ch"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_gen(ch=flux_ch, nqz=1)
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])

        self.add_loop("fluxloop", cfg["steps_flux"])
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
        self.add_pulse(
            ch=flux_ch,
            name="flux_pulse",
            style="const",
            length=cfg["flux_length"],
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["flux_ch"], name="flux_pulse", t=0)
        self.delay(cfg["saturate_times"])
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Resonator_onetone_flux:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False, yoko: str = None, DAC=None):
        if liveplot:
            if yoko is not None:
                self.liveplot_yoko(py_avg, yoko_currnet=yoko, yoko_inst=DAC)
            else:
                self.liveplot_hardware(py_avg)

        else:
            prog = SingleToneSpectroscopyProgram_yoko(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        pass

    async def liveplot_yoko(
        self,
        py_avg,
        yoko_value: np.ndarray,
        yoko_inst: str = None,
        mode: str = "current",
    ):
        """
        Async liveplot for YOKO current/voltage sweep with spectroscopy program.
        Uses asyn_run(mode="2D") for live plotting.
        """
        from .YOKOGS200 import YOKOGS200
        import pyvisa

        assert mode in ("current", "voltage"), "mode must be 'current' or 'voltage'"

        # Initialize YOKO instrument
        rm = pyvisa.ResourceManager()
        yoko = YOKOGS200(yoko_inst, rm)

        # Measurement program
        prog = SingleToneSpectroscopyProgram_yoko(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)  # y-axis
        self.iqdata = np.zeros((len(yoko_value), len(self.freqs)), dtype=complex)
        self.yoko_currnet = yoko_value

        # X-axis label (unit conversion)
        x_unit_base = "A" if mode == "current" else "V"
        x_vals_disp = auto_unit(yoko_value, x_unit_base)  # {'value': array, 'unit': 'mA' ...}
        x_label = f"Current ({x_vals_disp['unit']})" if mode == "current" else f"Voltage ({x_vals_disp['unit']})"

        # Storage for final normalized map
        z_final = None

        # Sweep through all YOKO setpoints
        for idx, setpoint in tqdm(enumerate(yoko_value), total=len(yoko_value), desc="yoko sweep"):
            # Configure YOKO
            if mode == "current":
                yoko.SetMode("current")
                yoko.SetCurrent(setpoint)
            else:
                yoko.SetMode("voltage")
                yoko.SetVoltage(setpoint)

            # Update title with current setpoint in proper unit
            sp_disp = auto_unit(setpoint, x_unit_base)
            title_str = (
                f"OneTone Current = {sp_disp['value']:.2f}{sp_disp['unit']} : {idx + 1}/{len(yoko_value)}"
                if mode == "current"
                else f"OneTone Voltage = {sp_disp['value']:.2f}{sp_disp['unit']} : {idx + 1}/{len(yoko_value)}"
            )

            # Initialize interactive heatmap
            z0 = np.zeros((len(self.freqs), idx + 1), dtype=float)
            figw = go.FigureWidget(
                data=[
                    go.Heatmap(
                        x=x_vals_disp["value"][: idx + 1],
                        y=self.freqs,
                        z=z0,
                        zmin=0.0,
                        zmax=1.0,
                        showscale=True,
                    )
                ]
            )
            figw.update_layout(
                title="OneTone Flux",
                xaxis_title=x_label,
                yaxis_title="Frequency (MHz)",
                autosize=True,
                margin=dict(l=10, r=10, t=30, b=30, pad=0),
            )
            figw.update_xaxes(constrain="domain")
            figw.update_yaxes(constrain="domain")
            display(figw)

            # Run acquisition + live plot (2D)
            iq_avg, z_map = await asyn_run(prog, self.soc, py_avg, figw, title=title_str, mode="2D")

            # Save data (for this setpoint)
            self.iqdata[idx, :] = iq_avg.mean(axis=0)  # collapse freq dimension
            z_final = z_map  # update last snapshot

            figw.close()

        # ===== Final static Heatmap (with slim colorbar, tight margins) =====
        fig_final = go.Figure(
            data=[
                go.Heatmap(
                    x=x_vals_disp["value"],
                    y=self.freqs,
                    z=z_final,
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(
                        thickness=12,
                        x=1.0,
                        xpad=4,
                        len=0.9,
                    ),
                )
            ]
        )
        fig_final.update_layout(
            title="Resonator Flux Spectroscopy",
            xaxis_title=x_label,
            yaxis_title="Frequency (MHz)",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        fig_final.update_xaxes(constrain="domain")
        fig_final.update_yaxes(constrain="domain")
        display(fig_final)

    async def liveplot_hardware(self, py_avg=1):
        """
        Async liveplot for hardware spectroscopy (flux vs freq 2D heatmap).
        Uses the new asyn_run() with mode="2D".
        """
        prog = SingleToneSpectroscopyProgram_hardware(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)

        # Initialize interactive heatmap
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
            title=f"average: 0 / {py_avg}",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Flux Gains",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        # Run async acquisition + plotting
        self.iqdata, z_final = await asyn_run(prog, self.soc, py_avg, figw,
                                            title="Resonator Flux Spectroscopy",
                                            mode="2D")

        # Close interactive widget
        figw.close()

        # Final static heatmap
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
            title="Resonator Flux Spectroscopy (final)",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Flux Gains",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        fig_final.update_xaxes(constrain="domain")
        fig_final.update_yaxes(constrain="domain")
        display(fig_final)

    def saveLabber(self, qb_idx, yoko_value=None, mode: str = "current"):
        expt_name = "s002_onetone_flux" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        try:
            self.cfg.pop("res_freq_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        if yoko_value is not None:
            if mode == "current":
                hdf5_generator(
                    filepath=file_path,
                    x_info={
                        "name": "Frequency",
                        "unit": "Hz",
                        "values": self.freqs * 1e6,
                    },
                    y_info={"name": "Yoko", "unit": "A", "values": yoko_value},
                    z_info={
                        "name": "Signal",
                        "unit": "ADC unit",
                        "values": self.iqdata,
                    },
                    comment=(f"{dict_val}"),
                    tag="OneTone",
                )
            elif mode == "voltage":
                hdf5_generator(
                    filepath=file_path,
                    x_info={
                        "name": "Frequency",
                        "unit": "Hz",
                        "values": self.freqs * 1e6,
                    },
                    y_info={"name": "Yoko", "unit": "V", "values": yoko_value},
                    z_info={
                        "name": "Signal",
                        "unit": "ADC unit",
                        "values": self.iqdata,
                    },
                    comment=(f"{dict_val}"),
                    tag="OneTone",
                )
        print(f"Data save to {file_path}")
