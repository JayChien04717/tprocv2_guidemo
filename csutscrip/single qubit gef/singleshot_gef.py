import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from expt_gef_cfg import config

from qick import *
# for now, all the tProc v2 classes need to be individually imported (can't use qick.*)
# the main program class
from qick.asm_v2 import AveragerProgramV2
# for defining sweeps
from qick.asm_v2 import QickSpan, QickSweep1D

import json
import datetime
import pprint as pp

# connect to zcu
import Pyro4
from qick import QickConfig
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

ns_host = "192.168.10.97"
ns_port = 8888
proxy_name = "myqick"

ns = Pyro4.locateNS(host=ns_host, port=ns_port)
soc = Pyro4.Proxy(ns.lookup(proxy_name))
soccfg = QickConfig(soc.get_cfg())


class SingleShotProgram_g(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        # no qubit pulse
        self.delay_auto(0.01, tag='wait')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleShotProgram_e(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_gain_ge'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(0.01, tag='wait')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class SingleShotProgram_f(AveragerProgramV2):
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

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['res_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['res_phase'],
                       gain=cfg['res_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp_ge",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_ge_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ge",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp_ef",
                       sigma=cfg['sigma'], length=cfg['sigma']*5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_ef_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ef",
                       freq=cfg['qubit_freq_ef'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_gain_ef'],
                       )

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_ge_pulse", t=0)  # play pulse
        self.delay_auto(0.01, tag='wait1')
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_ef_pulse", t=0)  # play pulse
        self.delay_auto(0.01)
        self.pulse(ch=self.cfg["qubit_ch"],
                   name="qubit_ge_pulse", t=0)  # play pulse
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


if __name__ == '__main__':
    # measure
    
    from singleshotplot import hist
    # expt_cfg={'steps': 5000}
    config.update([('steps', 5000)])
    ssp_g = SingleShotProgram_g(
        soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
    iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

    ssp_e = SingleShotProgram_e(
        soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
    iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

    ssp_f = SingleShotProgram_f(
        soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
    iq_list_f = ssp_f.acquire(soc, soft_avgs=1, progress=True)

    I_g = iq_list_g[0][0].T[0]
    Q_g = iq_list_g[0][0].T[1]
    I_e = iq_list_e[0][0].T[0]
    Q_e = iq_list_e[0][0].T[1]
    I_f = iq_list_f[0][0].T[0]
    Q_f = iq_list_f[0][0].T[1]

    # plot

    data = {
        'Ig':I_g,
        'Qg':Q_g,
        'Ie':I_e,
        'Qe':Q_e,
        'If':I_f,
        'Qf':Q_f,
    }


    hist(
        data,
        amplitude_mode=False,
        ps_threshold=None,
        theta=None,
        plot=True,
        verbose=True,
        fid_avg=False,
        fit=False,
        fitparams=None,
        normalize=True,
        title=None,
        export=False,
    )


    # # 假設這些變數是由其他部分提供的
    # expt_name = "Experiment"
    # QubitIndex = 1
    # DATA_PATH = "./data"  # 設置資料儲存目錄


    # # 建立實驗名稱
    # prefix = str(datetime.date.today())
    # exp_name = f"{expt_name}_Q{QubitIndex}_{prefix}"
    # print(f"Experiment name: {exp_name}")

    # # 生成檔案名稱
    # os.makedirs(DATA_PATH, exist_ok=True)
    # fname = get_next_filename(DATA_PATH, exp_name, suffix='.h5')
    # print(f"Current data file: {fname}")

    # # 將數據儲存到 HDF5 文件中
    # file_path = os.path.join(DATA_PATH, fname)
    # with h5py.File(file_path, 'a') as f:
    #     # 添加數據到文件
    #     f.create_dataset('I_g', data=I_g)
    #     f.create_dataset('Q_g', data=Q_g)
    #     f.create_dataset('I_e', data=I_e)
    #     f.create_dataset('Q_e', data=Q_e)
    #     f.create_dataset('I_f', data=I_f)
    #     f.create_dataset('Q_f', data=Q_f)

    #     # 將配置保存為屬性
    #     f.attrs['config'] = json.dumps(config)

    # print(f"Data saved to: {file_path}")