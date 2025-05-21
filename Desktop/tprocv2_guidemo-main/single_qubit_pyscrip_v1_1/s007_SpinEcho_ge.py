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
from .system_tool import  get_next_filename_labber, hdf5_generator
from .module_fitzcu import T2fring_analyze
from IPython.display import display, clear_output
##################
# Define Program #
##################


class SpinEchoProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        # pynq configured
        # self.declare_readout(ch=ro_ch, length=cfg['ro_len'], freq=cfg['f_res'], gen_ch=res_ch)

        # tproc configured
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_loop("waitloop", cfg["steps"])

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_pi2_gain_ge'],
                       )

        # pi pulse
        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_pi_gain_ge'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'] +
                       cfg['wait_time']*360*cfg['ramsey_freq'],
                       gain=cfg['qubit_pi2_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)
        self.delay_auto((cfg['wait_time']/2)+0.01, tag='wait1')
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse_pi", t=0)
        self.delay_auto((cfg['wait_time']/2)+0.01, tag='wait2')
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class SpinEcho:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg):
        prog = SpinEchoProgram(
            self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
        self.iq_list[0][0].dot([1,1j])
        self.delay_times = (prog.get_time_param('wait1', "t", as_array=True) +
                            prog.get_time_param('wait2', "t", as_array=True))

    def plot(self):
        T2fring_analyze(self.delay_times, self.iq_list[0][0].dot([1,1j]), prefix='Spin Echo')

    def liveplot(self, py_avg):
        iq = 0

        marker_style = {'marker': 'o', 'markersize': 5, 'alpha':0.7, 'linestyle': '-',}
        fig, ax = plt.subplots(figsize=(6, 4))
        prog = SpinEchoProgram(
            self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        self.delay_times = (prog.get_time_param('wait1', "t", as_array=True) +
                            prog.get_time_param('wait2', "t", as_array=True))

        for avg in tqdm(range(py_avg), desc='average count'):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)

            iq_data = self.iq_list[0][0].dot([1,1j])
            iq = iq_data if avg == 0 else iq + iq_data
            iq_avg = iq / (avg + 1)

            ax.cla()
            ax.plot(self.delay_times, np.abs(iq_avg), **marker_style)
            ax.set_title(f'average: {avg+1} / {py_avg}')
            ax.set_xlabel('Times (us)')
            ax.set_ylabel('Signal (ADC unit)')
            ax.grid(True)
            clear_output(wait=True)
            display(fig)

        plt.close(fig)

    def saveLabber(self, qb_idx):
        expt_name = "s007_SpinEcho_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name)

        hdf5_generator(
                filepath=file_path,
                x_info={'name': 'Times', 'unit': "us",
                        'values': self.delay_times},
                z_info={'name': 'Signal', 'unit': 'ADC unit',
                        'values':  self.iq_list[0][0].dot([1,1j])},
                comment=(),
                tag= 'Spin Echo'
        )

        print(f'Data save to {file_path}')

if __name__=="__main__":
    ###################
    # Experiment sweep parameter
    ###################

    START_TIME = 0.0  # [us]
    STOP_TIME = 100  # [us]
    STEPS = 100
    config.update([('steps', STEPS), ('wait_time',
                QickSweep1D('waitloop', START_TIME, STOP_TIME))])

    ###################
    # Run the Program
    ###################

    se = SpinEchoProgram(
        soccfg, reps=100, final_delay=config['relax_delay'], cfg=config)
    py_avg = 10
    iq_list = se.acquire(soc, soft_avgs=py_avg, progress=True)
    delay1 = se.get_time_param('wait1', "t", as_array=True)
    delay2 = se.get_time_param('wait2', "t", as_array=True)
    delay_times = delay1 + delay2
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
        exp_name = expt_name + '_Q' + str(QubitIndex)
        print('Experiment name: ' + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        print('Current data file: ' + file_path)

        data_dict = {
            "x_name": "Ramsey time(us)",
            "x_value": delay_times*2,

            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j])
        }

        result = {
            "T1": "350us",
            "T2": "130us"
        }

        saveh5(file_path, data_dict, result)
