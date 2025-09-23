
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
from .async_fun import asyn_run
# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import  get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from .module_fitzcu import  post_rotate
from .fitting import *
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################

class SingleToneSpectroscopyCoolingProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        cool_ch1 = cfg['cool_ch1']
        cool_ch2 = cfg['cool_ch2']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=ro_ch, length=cfg['ro_length'])

        if self.soccfg['gens'][cool_ch1]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=cool_ch1, nqz=cfg['nqz_cool_ch1'], mixer_freq=cfg['cool_mixer1'])
        else:
            self.declare_gen(ch=cool_ch1, nqz=cfg['nqz_cool_ch1'])

        if self.soccfg['gens'][cool_ch2]['type']=='axis_sg_int4_v2':
            self.declare_gen(ch=cool_ch2, nqz=cfg['nqz_cool_ch2'], mixer_freq=cfg['cool_mixer2'])
        else:
            self.declare_gen(ch=cool_ch2, nqz=cfg['nqz_cool_ch2'])

        self.add_loop("freqloop2", cfg["f_steps2"])
        self.add_loop("freqloop1", cfg["f_steps1"])
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'], gen_ch=res_ch)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )
        self.add_pulse(ch=cool_ch1, name="cool_pulse1", ro_ch=ro_ch,
                       style="const",
                       length=cfg['cool_length'],
                       freq=cfg['cool_freq_1'],
                       phase=0,
                       gain=cfg['cool_gain_1'],
                       )
        self.add_pulse(ch=cool_ch2, name="cool_pulse2", ro_ch=ro_ch,
                       style="const",
                       length=cfg['cool_length'],
                       freq=cfg['cool_freq_2'],
                       phase=0,
                       gain=cfg['cool_gain_2'],
                       )
    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg['cool_ch1'], name="cool_pulse1", t=0)
        self.pulse(ch=cfg['cool_ch2'], name="cool_pulse2", t=0)
        self.delay_auto(0.05, tag="Ring down")
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleToneSpectroscopyCooling:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            self.liveplot(py_avg)
        else:
            prog = SingleToneSpectroscopyCoolingProgram(
                self.soccfg, reps=self.cfg['reps'], final_delay=self.cfg['relax_delay'], cfg=self.cfg)

            self.iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = self.iq_list[0][0].dot([1, 1j])
            self.freqs1 = prog.get_pulse_param("cool_pulse1", "freq", as_array=True)
            self.freqs2 = prog.get_pulse_param("cool_pulse2", "freq", as_array=True)

    def plot(self):
        data = np.abs(post_rotate(self.iqdata))  # shape: (n_gain, n_freq)
        data_norm = np.array([
            (row - np.min(row)) / (np.max(row) - np.min(row)) if np.max(row) != np.min(row) else row
            for row in data
        ])
        pcm = plt.pcolormesh(self.freqs1, self.freqs2, data_norm)
        plt.title('Resonator Cooling')
        plt.ylabel(r'$|f,0\rangle - |g,1\rangle$')
        plt.xlabel(r'$|f,0\rangle - |e,0\rangle$')
        plt.colorbar(pcm)

    async def liveplot(self, py_avg=1):
        """
        Async liveplot for cooling spectroscopy with two frequency sweeps.
        Uses asyn_run(mode="2D") for live plotting.
        """
        prog = SingleToneSpectroscopyCoolingProgram(
            self.soccfg,
            reps=self.cfg['reps'],
            final_delay=self.cfg['relax_delay'],
            cfg=self.cfg,
        )
        self.freqs1 = prog.get_pulse_param("cool_pulse1", "freq", as_array=True)  # x-axis
        self.freqs2 = prog.get_pulse_param("cool_pulse2", "freq", as_array=True)  # y-axis

        # Initialize empty heatmap
        z0 = np.zeros((len(self.freqs2), len(self.freqs1)), dtype=float)
        figw = go.FigureWidget(
            data=[
                go.Heatmap(
                    x=self.freqs1,
                    y=self.freqs2,
                    z=z0,
                    zmin=0.0,
                    zmax=1.0,
                    showscale=True,
                )
            ]
        )
        figw.update_layout(
            title=f"average: 0 / {py_avg}",
            xaxis_title=r"$|f,0\rangle - |e,0\rangle$",
            yaxis_title=r"$|f,0\rangle - |g,1\rangle$",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        # Run async acquisition + live plot
        self.iqdata, z_final = await asyn_run(
            prog, self.soc, py_avg, figw,
            title="Resonator ge Cooling",
            mode="2D"
        )

        # Close interactive widget
        figw.close()

        # Final static heatmap
        fig_final = go.Figure(
            data=[
                go.Heatmap(
                    x=self.freqs1,
                    y=self.freqs2,
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
            title="Resonator ge Cooling (final)",
            xaxis_title=r"$|f,0\rangle - |e,0\rangle$",
            yaxis_title=r"$|f,0\rangle - |g,1\rangle$",
            autosize=True,
            margin=dict(l=10, r=10, t=30, b=30, pad=0),
        )
        fig_final.update_xaxes(constrain="domain")
        fig_final.update_yaxes(constrain="domain")
        display(fig_final)


    def saveLabber(self, qb_idx, yoko_current=None):
        expt_name = "002b_res_ge_punchout" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_current)
        try:
            self.cfg.pop('res_freq_ge')
            self.cfg.pop('res_gain_ge')
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
                filepath=file_path,
                x_info={'name': r'$|2,0\rangle - |0,1\rangle$', 'unit': "Hz",
                        'values': self.freqs1*1e6},
                y_info={'name': r'$|2,0\rangle - |1,0\rangle$', 'unit': "Hz",
                        'values': self.freqs2*1e6},
                z_info={'name': 'Signal', 'unit': 'ADC unit',
                        'values':  self.iqdata},
                comment=(f'{dict_val}'),
                tag= 'OneTone'
        )
        print(f'Data save to {file_path}')

if __name__ == '__main__':
    ###################
    # Experiment sweep parameter
    ###################


    # START_FREQ = 5000  # [MHz]
    # STOP_FREQ = 6000  # [MHz]
    # STEPS_freq = 100

    # START_gain = 0.1  # [MHz]
    # STOP_gain = 0.5  # [MHz]
    # STEPS_gain = 5
    # config.update([('f_steps', STEPS_freq), ('res_freq_ge', QickSweep1D('freqloop', START_FREQ, STOP_FREQ)),
    #             ('g_steps', STEPS_freq), ('res_gain_ge', QickSweep1D('gainloop', START_gain, STOP_gain))])

    # ###################
    # # Run the Program
    # ###################
    # punchout = SingleToneSpectroscopyPunchout(soccfg, config)
    # punchout.run(reps=1)
    # punchout.plot()
    pass