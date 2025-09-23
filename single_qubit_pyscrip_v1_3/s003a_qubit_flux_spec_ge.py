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
import pyvisa

# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator, auto_unit
from tqdm.auto import tqdm
from .fitting import *
from .yamltool import yml_comment
from .YOKOGS200 import YOKOGS200
from .async_fun import asyn_run
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

    async def liveplot_yoko(
        self,
        py_avg: int,
        yoko_value: np.ndarray,
        yoko_inst: str = None,
        mode: str = "current",
    ):
        """
        Async liveplot for Qubit two-tone spectroscopy with YOKO sweep.
        Sweeps current/voltage on YOKO and uses asyn_run(mode="1D") to
        acquire qubit spectrum at each setpoint.
        """

        assert mode in ("current", "voltage"), "mode must be 'current' or 'voltage'"

        # --- (Optional) YOKO init ---
        # rm = pyvisa.ResourceManager()
        # yoko = YOKOGS200(yoko_inst, rm)

        # --- Measurement program ---
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

        # --- Prepare storage ---
        self.iqdata = np.zeros((len(yoko_value), len(self.freqs)), dtype=complex)

        # --- Axis unit conversion ---
        x_unit_base = "A" if mode == "current" else "V"
        x_vals_disp = auto_unit(yoko_value, x_unit_base)  # {'value': array, 'unit': 'mA'...}
        x_label = (
            f"Current ({x_vals_disp['unit']})"
            if mode == "current"
            else f"Voltage ({x_vals_disp['unit']})"
        )

        # --- Initialize interactive heatmap ---
        z0 = np.zeros((len(self.freqs), len(yoko_value)), dtype=float)
        figw = go.FigureWidget(
            data=[
                go.Heatmap(
                    x=x_vals_disp["value"],
                    y=self.freqs,
                    z=z0,
                    zmin=0.0,
                    zmax=1.0,
                    showscale=True,
                    colorscale="Viridis",
                )
            ]
        )
        figw.update_layout(
            title=f"Qubit Two-tone sweep: 0/{len(yoko_value)}",
            xaxis_title=x_label,
            yaxis_title="Frequency (MHz)",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        # --- Sweep through YOKO setpoints ---
        for idx, sp in tqdm(enumerate(yoko_value), total=len(yoko_value), desc="YOKO sweep"):
            # (Optional) Configure YOKO
            if mode == "current":
                # yoko.SetMode("current"); yoko.SetCurrent(sp)
                sp_disp = auto_unit(sp, "A")
                title_str = (
                    f"Qubit Twotone Current = {sp_disp['value']:.2f}{sp_disp['unit']} "
                    f": {idx + 1}/{len(yoko_value)}"
                )
            else:
                # yoko.SetMode("voltage"); yoko.SetVoltage(sp)
                sp_disp = auto_unit(sp, "V")
                title_str = (
                    f"Qubit Twotone Voltage = {sp_disp['value']:.2f}{sp_disp['unit']} "
                    f": {idx + 1}/{len(yoko_value)}"
                )

            # --- Run async acquisition (1D spectrum) ---
            iq_avg = await asyn_run(prog, self.soc, py_avg, figw, title=title_str, mode="1D")

            # Save data
            self.iqdata[idx, :] = iq_avg

            # Normalize globally (across all finished setpoints)
            amp = np.abs(self.iqdata[: idx + 1, :]).T  # (n_freq, n_x_done)
            a_min, a_max = np.nanmin(amp), np.nanmax(amp)
            z_norm = (amp - a_min) / (a_max - a_min) if a_max > a_min else np.zeros_like(amp)

            # Update heatmap
            with figw.batch_update():
                figw.data[0].z[:, : idx + 1] = z_norm
                figw.layout.title.text = title_str

        # --- Close interactive widget ---
        figw.close()

        # --- Final static heatmap ---
        fig_final = go.Figure(
            data=[
                go.Heatmap(
                    x=x_vals_disp["value"],
                    y=self.freqs,
                    z=z_norm,
                    zmin=0.0,
                    zmax=1.0,
                    colorscale="Viridis",
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
            title="Qubit Twotone Flux (final)",
            xaxis_title=x_label,
            yaxis_title="Frequency (MHz)",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        fig_final.update_xaxes(constrain="domain")
        fig_final.update_yaxes(constrain="domain")
        display(fig_final)

        return self.iqdata, z_norm



    def saveLabber(self, qb_idx, yoko_value=None, mode: str = "current"):
        expt_name = "003_qubit_flux_spec_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        try:
            self.cfg.pop("qubit_freq_ge")
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
                    tag="TwoTone",
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
                    tag="TwoTone",
                )
        print(f"Data save to {file_path}")
