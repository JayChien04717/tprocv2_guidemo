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
from .system_tool import  get_next_filename_labber, hdf5_generator
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
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        # self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        if self.soccfg['gens'][qubit_ch]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'], mixer_freq=cfg['qmixer_freq'])
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        self.add_loop("freqloop", cfg["steps"])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=0,
                       gain=cfg['qubit_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(0.05)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class Qubit_Twotone:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = PulseProbeSpectroscopyProgram(
                self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1,1j])
            self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

    def plot(self):
        f_q = spectrum_analyze(self.freqs, self.iqdata)
        return f_q


    def liveplot(self, py_avg):
        iq = 0
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

        marker_style = {'marker': 'o', 'markersize': 5, 'alpha':0.7, 'linestyle': '-',}
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in tqdm(range(py_avg), desc='average count'):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)


            ax.cla()
            ax.plot(self.freqs, np.abs(post_rotate(self.iqdata)), **marker_style)
            ax.set_title(f'average: {i+1} / {py_avg}')
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('ADC unist')
            ax.grid(True)
            clear_output(wait=True)
            display(fig)
        clear_output(wait=True)
        ax.set_title(f'Qubit ge Spectrum')
        ax.plot(self.freqs, np.abs(post_rotate(self.iqdata)), **marker_style)
        pOpt, _ = fitlor(self.freqs, np.abs(post_rotate(self.iqdata)))
        res = pOpt[2]  # Extract resonance frequency

        ax.plot(self.freqs, lorfunc(self.freqs, *pOpt), label='Fit')
        ax.axvline(res, color='r', ls='--',
                        label=f'$f_{{res}}$ = {res:.2f} MHz')
        ax.legend()
        self.sim = lorfunc(self.freqs, *pOpt)
        return round(res,4)

    def saveLabber(self, qb_idx, yoko_current=None, save_sim=False):
        expt_name = "003_qubit_spec_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop('qubit_freq_ge')
        except:
            pass

        dict_val = yml_comment(self.cfg)
        if save_sim:
            hdf5_generator(
                    filepath=file_path,
                    x_info={'name': 'Frequency', 'unit': "Hz",
                            'values': self.freqs*1e6},
                    y_info={'name': 'simulate', 'unit': "None",
                            'values': np.array([0,1])},
                    z_info={'name': 'Signal', 'unit': 'ADC unit',
                            'values':  np.array([self.iqdata, self.sim])},
                    comment=(f'{dict_val}'),
                    tag= 'TwoTone'
            )
        else:
            hdf5_generator(
                    filepath=file_path,
                    x_info={'name': 'Frequency', 'unit': "Hz",
                            'values': self.freqs*1e6},  

                    z_info={'name': 'Signal', 'unit': 'ADC unit',
                            'values':  self.iqdata},
                    comment=(f'{dict_val}'),
                    tag= 'TwoTone'
            )
        print(f'Data save to {file_path}')

        
if __name__=='__main':

    ###################
    # Run the Program
    ###################

    START_FREQ = 4000  # [MHz]
    STOP_FREQ = 6000  # [MHz]
    STEPS = 101
    config.update([('steps', STEPS), ('qubit_freq_ge',
                QickSweep1D('freqloop', START_FREQ, STOP_FREQ))])

    ###################
    # Run the Program
    ###################

    qspec = PulseProbeSpectroscopyProgram(
        soccfg, reps=10, final_delay=0.5, cfg=config)
    py_avg = config['py_avg']
    iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
    freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)
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
        exp_name = expt_name + '_Q' + str(QubitIndex)
        print('Experiment name: ' + exp_name)
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        print('Current data file: ' + file_path)

        data_dict = {
            "x_name": "x_axis",
            "x_value": freqs,

            "z_name": "iq_list",
            "z_value": iq_list[0][0].dot([1, 1j])
        }

        result = {
            "T1": "350us",
            "T2": "130us"
        }

        saveh5(file_path, data_dict, result)