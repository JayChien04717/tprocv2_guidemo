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
from .module_fitzcu import T1_analyze, post_rotate
from .fitting import expfunc, fitexp
from .yamltool import yml_comment
from IPython.display import display, clear_output
##################
# Define Program #
##################


class T1Program(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])

        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(
                ch=qubit_ch, nqz=cfg["nqz_qubit"], mixer_freq=cfg["qmixer_freq"]
            )
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("waitloop", cfg["steps"])

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

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse",
            ro_ch=ro_ch,
            style="arb",
            envelope="ramp",
            freq=cfg["qubit_freq_ge"],
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_pi_gain_ge"],
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

        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class T1:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)

        else:
            prog = T1Program(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iq_list[0][0].dot([1, 1j])
            self.delay_times = prog.get_time_param("wait", "t", as_array=True)

    def plot(self):
        T1_analyze(self.delay_times, self.iq_list[0][0].dot([1, 1j]))

    def liveplot(self, py_avg):
        iq = 0

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        prog = T1Program(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)

        for avg in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)

            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if avg == 0 else iq + iq_data
            self.iqdata = iq / (avg + 1)

            ax.cla()
            ax.plot(self.delay_times, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f"average: {avg + 1} / {py_avg}")
            ax.set_xlabel("Times (us)")
            ax.set_ylabel("Signal (ADC unit)")
            ax.grid(True)
            clear_output(wait=True)
            display(fig)

        clear_output(wait=True)

        pOpt, pCov = fitexp(self.delay_times, np.abs(post_rotate(self.iqdata)))
        p_sigma = np.sqrt(np.diag(pCov))
        ax.plot(self.delay_times, expfunc(self.delay_times, *pOpt), label="Fit")
        ax.set_title(f"T1 = {pOpt[3]:.2f} +-{p_sigma[3]:.2f}$\mu s$", fontsize=15)
        ax.legend()
        self.sim = expfunc(self.delay_times, *pOpt)

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "s008_T1_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)

        try:
            self.cfg.pop("wait_time")
        except:
            pass

        dict_val = yml_comment(self.cfg)
        if save_sim:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Times", "unit": "us", "values": self.delay_times},
                # y_info={'name': 'simulate', 'unit': "None",
                #         'values': np.array([0,1])},
                z_info={
                    "name": "Signal",
                    "unit": "ADC unit",
                    "values": np.array([self.iqdata, self.sim]),
                },
                comment=(f"{dict_val}"),
                tag="T1",
            )
        else:
            hdf5_generator(
                filepath=file_path,
                x_info={"name": "Times", "unit": "us", "values": self.delay_times},
                z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
                comment=(f"{dict_val}"),
                tag="T1",
            )

        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Experiment sweep parameter
    ###################

    START_TIME = 0.0  # [us]
    STOP_TIME = 100  # [us]
    STEPS = 100
    config.update(
        [
            ("steps", STEPS),
            ("wait_time", QickSweep1D("waitloop", START_TIME, STOP_TIME)),
        ]
    )

    ###################
    # Run the Program
    ###################

    t1 = T1Program(soccfg, reps=100, final_delay=config["relax_delay"], cfg=config)
    py_avg = 10
    iq_list = t1.acquire(soc, soft_avgs=py_avg, progress=True)
    delay_times = t1.get_time_param("wait", "t", as_array=True)
    amps = np.abs(iq_list[0][0].dot([1, 1j]))

    ###################
    # Plot
    ###################

    Plot = True

    if Plot:
        # plt.plot(freqs,  iq_list[0][0].T[0])
        # plt.plot(freqs,  iq_list[0][0].T[1])
        plt.plot(delay_times, iq_list[0][0].dot([1, 1j]))
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
            "x_name": "Ramsey time(us)",
            "x_value": delay_times * 2,
            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j]),
        }

        result = {"T1": "350us", "T2": "130us"}

        saveh5(file_path, data_dict, result)
