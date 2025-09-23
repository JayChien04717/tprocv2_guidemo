# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from tqdm.auto import tqdm

# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator
from .module_fitzcu import amprabi_analyze
from .fitting import decaysin, fitdecaysin, fix_phase
from .plot_utils import plot_final2_plotly
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class AmplitudeRabiProgram(AveragerProgramV2):
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

        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        self.add_loop("gainloop", cfg["steps"])

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

        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        if cfg["qubit_ge_pulse_style"] == "arb":
            self.add_pulse(
                ch=qubit_ch,
                name="qubit_pulse",
                style="arb",
                envelope="ramp",
                freq=cfg["qubit_freq_ge"],
                phase=cfg["qubit_phase"],
                gain=cfg["qubit_gain_ge"],
            )
        elif cfg["qubit_ge_pulse_style"] == "flat_top":
            if cfg["qubit_flat_top_length_ge"] is None:
                raise ValueError("Please set qubit_flat_top_length_ge in config")
            self.add_pulse(
                ch=qubit_ch,
                name="qubit_pulse",
                style="flat_top",
                envelope="ramp",
                freq=cfg["qubit_freq_ge"],
                phase=cfg["qubit_phase"],
                gain=cfg["qubit_gain_ge"],
                length=cfg["qubit_flat_top_length_ge"],
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
            name="cooling1",
            sigma=cfg["res_sigma"],
            length=cfg["res_sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch1,
            name="cool_pulse1",
            envelope="cooling1",
            style="flat_top",
            length=cfg["cool_length"],
            freq=cfg["cool_freq_1"],
            phase=0,
            gain=cfg["cool_gain_1"],
        )
        self.add_gauss(
            ch=cool_ch2,
            name="cooling2",
            sigma=cfg["res_sigma"],
            length=cfg["res_sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=cool_ch2,
            name="cool_pulse2",
            envelope="cooling2",
            style="flat_top",
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

        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


class Amp_Rabi:
    def __init__(self, soc, soccfg, config):
        self.soc = soc
        self.soccfg = soccfg
        self.cfg = config

    def run(self, py_avg, liveplot=False):
        if liveplot:
            return self.liveplot(py_avg)
        else:
            prog = AmplitudeRabiProgram(
                self.soccfg,
                reps=self.cfg["reps"],
                final_delay=self.cfg["relax_delay"],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(self.soc, soft_avgs=py_avg, progress=True)
            self.iqdata = iq_list[0][0].dot([1, 1j])
            self.gains = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)

    def plot(self):
        pi_gain, pi2_gain = amprabi_analyze(self.gains, self.iqdata)
        return pi_gain, pi2_gain

    def liveplot(self, py_avg):
        iq = 0

        # ----- 量測程式 -----
        prog = AmplitudeRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.gains = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)

        # 避免 py_avg == 0 時出錯
        self.iqdata = np.zeros_like(self.gains, dtype=complex)

        # ===== 互動折線圖 (FigureWidget) =====
        figw = go.FigureWidget(
            data=[
                go.Scatter(
                    x=self.gains,
                    y=np.zeros_like(self.gains, dtype=float),
                    mode="lines+markers",
                    name="Amplitude",
                    showlegend=False,  # 不顯示 legend
                )
            ]
        )
        figw.update_layout(
            title=dict(
                text=f"average: 0 / {py_avg}",
                x=0.5,
                xanchor="center",
                y=0.95,
                yanchor="top",
            ),
            xaxis_title="DAC Gain (a.u.)",
            yaxis_title="ADC units",
            autosize=True,
            margin=dict(l=10, r=10, t=50, b=30, pad=0),
            showlegend=False,  # 全域關閉 legend
        )
        figw.update_xaxes(constrain="domain")
        figw.update_yaxes(constrain="domain")
        display(figw)

        # ===== live 更新 =====
        for i in tqdm(range(py_avg), desc="average count"):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            with figw.batch_update():
                figw.data[0].x = self.gains
                figw.data[0].y = np.abs(self.iqdata)
                figw.layout.title.text = f"average: {i + 1} / {py_avg}"

        # Live 結束，標題改為最終摘要
        with figw.batch_update():
            figw.layout.title = dict(
                text="Power Rabi ge", x=0.5, xanchor="center", y=0.95, yanchor="top"
            )
        figw.close()

        # ===== 最終摘要圖 (pure Plotly) =====
        # 這個函式需與你的頻譜版本一致：回傳 (fit_params, error, fig)
        fit_params, error, fig = plot_final2_plotly(
            self.gains, np.abs(self.iqdata), "DAC Gain (a.u.)", fitdecaysin, decaysin
        )

        # 計算 pi / pi2 位置
        pi_gain, pi2_gain = fix_phase(fit_params)

        title_text = f"Power Rabi ge — π gain = {pi_gain:.6f}, π/2 = {pi2_gain:.6f}"
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor="center", y=0.95, yanchor="top"),
            autosize=True,
            margin=dict(l=10, r=10, t=60, b=30, pad=0),
            showlegend=False,  # 不顯示 legend
        )

        # 軸域保持緊湊
        fig.update_xaxes(constrain="domain", title="DAC Gain (a.u.)")
        fig.update_yaxes(constrain="domain", title="ADC units")

        # 加上 π 與 π/2 的垂直線（不進 legend）
        fig.add_vline(x=pi_gain, line_dash="dash", line_width=2)
        fig.add_vline(x=pi2_gain, line_dash="dash", line_width=2)

        display(fig)

        # ===== (選擇性) 存檔 =====
        # import os, plotly.io as pio
        # fig_dir = os.path.join(DATA_PATH, "Fig")
        # os.makedirs(fig_dir, exist_ok=True)
        # pio.write_html(fig, file=os.path.join(fig_dir, "power_rabi_ge.html"), auto_open=False, include_plotlyjs="cdn")
        # 若要輸出 PNG，需要安裝 kaleido：pip install -U kaleido
        # pio.write_image(fig, os.path.join(fig_dir, "power_rabi_ge.png"), scale=2)

        return round(pi_gain, 6), round(pi2_gain, 6)

    def saveLabber(self, qb_idx, yoko_value=None):
        expt_name = "s005_power_rabi_ge" + f"_Q{qb_idx}"
        file_path = get_next_filename_labber(DATA_PATH, expt_name, yoko_value)
        try:
            self.cfg.pop("qubit_gain_ge")
        except:
            pass

        dict_val = yml_comment(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Gain", "unit": "DAC unit", "values": self.gains},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=(f"{dict_val}"),
            tag="Rabi",
        )

        print(f"Data save to {file_path}")
