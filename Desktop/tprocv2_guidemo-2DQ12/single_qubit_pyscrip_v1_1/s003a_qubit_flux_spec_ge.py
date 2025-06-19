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

        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse",
            ro_ch=ro_ch,
            style="const",
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


class Qubit_Twotone_Flux:
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

    def liveplot_yoko(self, py_avg, yoko_currnet: np.ndarray, yoko_inst: str = None):
        from .YOKOGS200 import YOKOGS200
        import pyvisa

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
        self.iqdata = np.zeros((len(yoko_currnet), len(self.freqs)))
        self.yoko_currnet = yoko_currnet
        mesh = ax.pcolormesh(
            yoko_currnet * 1e3,
            self.freqs,
            self.iqdata.T,  # transpose: pcolormesh expects (Y, X)
            shading="nearest",
        )

        ax.set_ylabel("Frequency (MHz)")
        ax.set_xlabel("Current (mA)")

        for idx, curr in tqdm(enumerate(yoko_currnet)):
            yoko.SetCurrent(curr)
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
            iq = iq_list[0][0].dot([1, 1j])
            self.iqdata[idx, :] = np.abs(iq)
            mesh.set_array(self.iqdata.T.ravel())
            mesh.set_clim(vmin=np.min(self.iqdata), vmax=np.max(self.iqdata))
            ax.set_title(
                f"Qubit Flux spec current ={curr * 1e3:.3f}mA : {idx + 1}/{len(yoko_currnet)}"
            )
            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)
        ax.pcolormesh(yoko_currnet * 1e3, self.freqs, self.iqdata.T, shading="nearest")
        ax.set_title("Qubit Twotone Flux")

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


if __name__ == "__main":
    ###################
    # Run the Program
    ###################

    START_FREQ = 4000  # [MHz]
    STOP_FREQ = 6000  # [MHz]
    STEPS = 101
    config.update(
        [
            ("steps", STEPS),
            ("qubit_freq_ge", QickSweep1D("freqloop", START_FREQ, STOP_FREQ)),
        ]
    )

    ###################
    # Run the Program
    ###################

    qspec = PulseProbeSpectroscopyProgram(soccfg, reps=10, final_delay=0.5, cfg=config)
    py_avg = config["py_avg"]
    iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
    freqs = qspec.get_pulse_param("qubit_pulse", "freq", as_array=True)
    amps = np.abs(iq_list[0][0].dot([1, 1j]))

    ###################
    # Plot
    ###################
    Plot = True

    if Plot:
        # plt.plot(freqs,  iq_list[0][0].T[0])
        # plt.plot(freqs,  iq_list[0][0].T[1])
        plt.plot(freqs, amps)
        plt.show()

    #####################################
    # ----- Saves data to a file ----- #
    #####################################
    Save = True
    if Save:
        data_path = "./data"
        labber_data = "./data/Labber"
        exp_name = expt_name + "_Q" + str(QubitIndex)
        print("Experiment name: " + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix=".h5")
        print("Current data file: " + file_path)

        data_dict = {
            "x_name": "x_axis",
            "x_value": freqs,
            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j]),
        }

        result = {"T1": "350us", "T2": "130us"}

        saveh5(file_path, data_dict, result)
