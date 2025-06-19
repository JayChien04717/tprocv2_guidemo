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
from .system_tool import hdf5_generator, get_next_filename_labber


##################
# Define Program #
##################


class LoopbackProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['freq'], gen_ch=gen_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_pulse(
            ch=res_ch,
            name="myconst",
            ro_ch=ro_ch,
            style="const",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="myconst", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0)


class TOF:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self):
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        self.iq_list = prog.acquire_decimated(self.soc, soft_avgs=self.cfg["soft_avgs"])
        self.t = prog.get_time_axis(ro_index=0)

    def plot(self):
        if self.iq_list is not None:
            plt.plot(self.t, self.iq_list[0].T[0])
            plt.plot(self.t, self.iq_list[0].T[1])
            plt.plot(self.t, np.abs((self.iq_list[0]).dot([1, 1j])))
            plt.xlabel("Time (us)")
            plt.ylabel("a.u")
            # plt.title("Time Of Flight")

            mean = np.mean(np.abs(self.iq_list[0].dot([1, 1j])))
            plt.axvline(
                self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > 1.5 * mean)],
                c="r",
                ls="--",
            )
            plt.title(
                f"Time of Flight Experiment, trig = {round(self.t[np.argmax(np.abs(self.iq_list[0].dot([1, 1j])) > 1.5 * mean)], 2)} us"
            )

        else:
            print("No data to plot. Run the experiment first.")

    def saveLabber(self, qb_idx):
        expt_name = "s001_tof" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.t * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "ADC unit",
                "values": self.iq_list[0].dot([1, 1j]),
            },
            comment=(),
            tag="OneTone",
        )
        print(f"Data save to {file_path}")


if __name__ == "__main__":
    ###################
    # Run the Program
    ###################
    tof = TOF(soccfg, config)
    tof.run()
    tof.plot()
    tof.save()
