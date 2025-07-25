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
from .module_fitzcu import resonator_circlefit, resonator_analyze, post_rotate
from .fitting import fit_asym_lor, asym_lorfunc
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class SingleToneSpectroscopyProgram(AveragerProgramV2):
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
            name="cool1",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            style="flat_top",
            envelope="cool1",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cool2",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            style="flat_top",
            envelope="cool2",
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
            self.delay_auto(0.05, tag="Ring down")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Resonator_onetone:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False, solve_type="hm"):
        if liveplot:
            return self.liveplot(py_avg, solve_type=solve_type)

        else:
            prog = SingleToneSpectroscopyProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )

            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        param = resonator_analyze(self.freqs, self.iqdata)
        return param

    def plot_circle(self):
        param = resonator_circlefit(self.freqs, self.iqdata)
        return param

    def liveplot(self, py_avg, solve_type="hm"):
        iq = 0
        prog = SingleToneSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            ax.cla()
            ax.plot(self.freqs, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f"average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("ADC unist")

            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        plt.close(fig)

        pOpt, _ = fit_asym_lor(
            self.freqs, np.abs(post_rotate(self.iqdata))
        )  # Fit the data
        param = resonator_circlefit(self.freqs, self.iqdata, solve_type=solve_type)
        self.sim = asym_lorfunc(self.freqs, *pOpt)
        return param

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "s002_onetone" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop("res_freq_ge")
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
                tag="OneTone",
            )
        else:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
                z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
                comment=(),
                tag="OneTone",
            )
        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Experiment sweep parameter
    ###################

    # START_FREQ = 4000  # [MHz]
    # STOP_FREQ = 5000  # [MHz]
    # STEPS = 101
    # config.update([('steps', STEPS), ('res_freq_ge',
    #             QickSweep1D('freqloop', START_FREQ, STOP_FREQ))])

    # ###################
    # # Run the Program
    # ###################

    # onetone = Resonator_onetone(soccfg, config)
    # onetone.run(reps=1)
    # onetone.plot()
    # onetone.save()
    pass
