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
from .module_fitzcu import lengthrabi_analyze
from IPython.display import display, clear_output
##################
# Define Program #
##################


class LengthRabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_gen(ch=qubit_ch, nqz=cfg["nqz_qubit"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
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
            phase=cfg["qubit_phase"],
            gain=cfg["qubit_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0.05, tag="waiting")

        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Time_Rabi:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, Start_time, Stop_time, Step):
        self.time_step = np.linspace(Start_time, Stop_time, Step)

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

    def liveplot(self, py_avg, Start_time, Stop_time, Step):
        iq = 0
        self.time_step = np.linspace(Start_time, Stop_time, Step)

        marker_style = {
            "marker": "o",
            "markersize": 5,
            "alpha": 0.7,
            "linestyle": "-",
        }

        fig, ax = plt.subplots(figsize=(6, 4))

        for avg in tqdm(range(py_avg), desc="average count"):
            iqlst = []
            for i in self.time_step:
                self.cfg["qubit_length_ge"] = i
                prog = LengthRabiProgram(
                    self.soccfg,
                    reps=self.cfg["reps"],
                    final_delay=self.cfg["relax_delay"],
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
                iqlst.append(iq_list[0][0].dot([1, 1j]))
            self.iqlst = np.array(iqlst)

            iq_data = self.iqlst
            iq = iq_data if avg == 0 else iq + iq_data
            iq_avg = iq / (avg + 1)

            ax.cla()
            ax.plot(self.time_step, np.abs(iq_avg), **marker_style)
            ax.set_title(f"average: {avg + 1} / {py_avg}")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit")

            clear_output(wait=True)
            display(fig)

        plt.close(fig)

    def saveLabber(self, qb_idx):
        expt_name = "004_time_rabi_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.time_step * 1e-6},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqlst},
            comment=(),
            tag="Rabi",
        )

        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Experiment sweep parameter
    ###################
    START_LEN = 0.01  # [us]
    STOP_LEN = 5  # [us]
    STEPS = 100
    qubit_pulse_len = np.linspace(START_LEN, STOP_LEN, STEPS)

    ###################
    # Run the Program
    ###################

    iq = []
    for i in tqdm(qubit_pulse_len):
        config["qubit_length_ge"] = i
        prog = LengthRabiProgram(
            soccfg, reps=config["reps"], final_delay=config["relax_delay"], cfg=config
        )
        py_avg = config["py_avg"]
        iq_list = prog.acquire(soc, soft_avgs=py_avg, progress=False)
        iq.append(iq_list[0][0].dot([1, 1j]))

    ###################
    # Plot
    ###################
    Plot = True

    if Plot:
        # plt.plot(freqs,  iq_list[0][0].T[0])
        # plt.plot(freqs,  iq_list[0][0].T[1])
        plt.plot(qubit_pulse_len, np.abs(iq))
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
            "x_value": qubit_pulse_len,
            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j]),
        }

        result = {"T1": "350us", "T2": "130us"}

        saveh5(file_path, data_dict, result)
