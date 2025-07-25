# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator


from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class SingleToneSpectroscopyProgram_yoko(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
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
        if yoko is not None:
            self.liveplot_yoko(py_avg, yoko_currnet=yoko, yoko_inst=DAC)
        if liveplot:
            self.liveplot(py_avg)

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

    def liveplot_yoko(
        self,
        py_avg,
        yoko_value: np.ndarray,
        yoko_inst: str = None,
        mode: str = "current",
    ):
        from .YOKOGS200 import YOKOGS200
        import pyvisa

        rm = pyvisa.ResourceManager()
        yoko = YOKOGS200(yoko_inst, rm)
        fig, ax = plt.subplots(figsize=(6, 4))
        prog = SingleToneSpectroscopyProgram_yoko(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.iqdata = np.zeros((len(yoko_value), len(self.freqs)))
        self.yoko_currnet = yoko_value
        mesh = ax.pcolormesh(
            yoko_value * 1e3,
            self.freqs,
            self.iqdata.T,  # transpose: pcolormesh expects (Y, X)
            shading="nearest",
        )

        ax.set_ylabel("Frequency (MHz)")

        for idx, curr in tqdm(enumerate(yoko_value)):
            if mode == "current":
                yoko.SetMode("current")
                yoko.SetCurrent(curr)
                ax.set_title(
                    f"Resonator Onetone Current ={curr * 1e3:.3f}mA : {idx + 1}/{len(yoko_value)}"
                )
                ax.set_xlabel("Current (mA)")
            elif mode == "voltage":
                yoko.SetMode("voltage")
                yoko.SetVoltage(curr)
                ax.set_title(
                    f"Resonator Onetone Voltage ={curr * 1e3:.3f}mV : {idx + 1}/{len(yoko_value)}"
                )
                ax.set_xlabel("Voltage (mV)")
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
            iq = iq_list[0][0].dot([1, 1j])
            self.iqdata[idx, :] = np.abs(iq)
            mesh.set_array(self.iqdata.T.ravel())
            mesh.set_clim(vmin=np.min(self.iqdata), vmax=np.max(self.iqdata))

            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)
        ax.pcolormesh(yoko_value * 1e3, self.freqs, self.iqdata.T, shading="nearest")
        if mode == "current":
            ax.set_title("Resonator Onetone Current")
        elif mode == "voltage":
            ax.set_title("Resonator Onetone Voltage")

    def liveplot_hardwre(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyProgram_hardware(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)

        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            ax.cla()
            ax.pcolormesh(self.freqs, self.gains, np.abs(self.iqdata))
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Flux Gains")

            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        ax.pcolormesh(self.freqs, self.gains, np.abs(self.iqdata))

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
