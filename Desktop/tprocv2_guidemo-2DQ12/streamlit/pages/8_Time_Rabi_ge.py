import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import datetime

from single_qubit_pyscrip.system_tool import (
    select_config_idx,
    saveh5,
    get_next_filename,
    format_time,
    hdf5_generator,
    get_next_filename_labber,
)
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.sidebar_config import (
    show_sidebar_config,
    config_update_sidebar,
    sync_param_to_config,
)
from single_qubit_pyscrip.SQ004_time_rabi_ge import LengthRabiProgram
from qick.asm_v2 import QickSweep1D

# -------------------------- Initialization -------------------------- #
st.title("Time Rabi ge")
st.session_state.expt_name = "004_time_rabi_ge"
Qubit = "Q" + str(st.session_state.QubitIndex)

st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.cooling_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

for key in ["timerabi", "config"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------------- Helper Functions -------------------------- #
def pipulse_analyze(pOpt):
    # 依據 fitting 結果自動算 pi/2, pi pulse 長度
    if pOpt[2] > 180:
        pOpt[2] -= 360
    elif pOpt[2] < -180:
        pOpt[2] += 360
    if pOpt[2] < 0:
        pi = (1 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2 = (0 - pOpt[2] / 180) / 2 / pOpt[1]
    else:
        pi = (3 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
        pi2 = (1 - pOpt[2] / 180) / 2 / pOpt[1]
    return pi, pi2


# -------------------------- Experiment Class -------------------------- #
class TimeRabi:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.soc = st.session_state.soc
        self.length = None
        self.iq_list = None
        self.iqdata = None

    def run(self, py_avg, fit=False):
        iq = 0
        prog = LengthRabiProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.length = prog.get_pulse_param("qubit_pulse", "length", as_array=True)
        marker_style = {"marker": "o", "markersize": 5, "alpha": 0.7, "linestyle": "-"}
        fig, ax = plt.subplots(figsize=(8, 4))
        placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = datetime.datetime.now()
        for i in range(py_avg):
            self.iq_list = prog.acquire(self.soc, soft_avgs=1, progress=False)
            iq_data = self.iq_list[0][0].dot([1, 1j])
            iq = iq_data if i == 0 else iq + iq_data
            self.iqdata = iq / (i + 1)

            ax.cla()
            ax.plot(self.length, np.abs(self.iqdata), **marker_style, label="Magnitude")
            ax.set_title(f"Average: {i + 1} / {py_avg}")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Signal (ADC unit)")

            placeholder.pyplot(fig)

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (py_avg - (i + 1))
            status_text.markdown(
                f"**Estimated time remaining: {format_time(remaining)}**"
            )
            progress_bar.progress((i + 1) / py_avg)
        placeholder.empty()
        plt.close(fig)
        if self.iq_list is not None:
            st.session_state.timetag = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    def save(self):
        path = get_next_filename(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
            suffix=".h5",
        )
        data_dict = {
            "experiment_name": "time_rabi_ge",
            "x_name": "Time (us)",
            "x_value": self.length,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j]),
        }
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveh5(path, data_dict, result=result_dict)

    def save_labber(self):
        if self.length is None or self.iq_list is None:
            st.error("No data available. Run the experiment first.")
            return
        path = get_next_filename_labber(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
        )
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=path,
            x_info={"name": "Time", "unit": "s", "values": self.length * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": self.iq_list[0][0].dot([1, 1j]),
            },
            comment=result_dict["notes"],
            tag="Rabi",
        )


# -------------------------- Parameters -------------------------- #
col1, col2, col3 = st.columns(3)
with col1:
    start_len = st.number_input("Start Length (us)", min_value=0.0, value=0.1, step=0.1)
with col2:
    stop_len = st.number_input(
        "Stop Length (us)", min_value=start_len, value=1.0, step=0.1
    )
with col3:
    steps = st.number_input("Steps", min_value=1, max_value=1000, value=101, step=1)

st.session_state.config.update(
    {
        "steps": steps,
        "qubit_length_ge": QickSweep1D("lenloop", start_len, stop_len),
    }
)

col_spacer, col_right = st.columns([2, 1])
with col_spacer:
    pyavg = st.number_input(
        "Soft average #", min_value=1, max_value=10000, value=10, step=1
    )
with col_right:
    st.header("Optional Parameters")
    qubit_gain = st.number_input(
        "Qubit Gain", min_value=1e-5, max_value=1.0, value=0.1, step=1e-5
    )
    sync_param_to_config("qubit_gain_ge", qubit_gain, target_cfg_group="qubit_cfg")
    sigma = st.number_input(
        "sigma", min_value=0.002, max_value=1.0, value=0.1, step=0.001
    )
    sync_param_to_config("sigma", sigma, target_cfg_group="qubit_cfg")
    relax_delay = st.number_input(
        "Relaxation time (us)", min_value=1, max_value=1000, value=10, step=1
    )
    st.session_state.config["relax_delay"] = relax_delay

# -------------------------- Sidebar -------------------------- #
qubit_index = int(st.session_state.get("QubitIndex", 1))
show_sidebar_config(
    st.session_state.hw_cfg,
    select_config_idx(st.session_state.readout_cfg, idx=qubit_index),
    select_config_idx(st.session_state.qubit_cfg, idx=qubit_index),
    select_config_idx(st.session_state.cooling_cfg, idx=qubit_index),
    st.session_state.expt_cfg,
    qubit_index=qubit_index,
)
config_update_sidebar(
    st.session_state.config,
    {
        "hw_cfg": st.session_state.hw_cfg,
        "readout_cfg": st.session_state.readout_cfg,
        "qubit_cfg": st.session_state.qubit_cfg,
        "cooling_cfg": st.session_state.cooling_cfg,
        "expt_cfg": st.session_state.expt_cfg,
    },
)

# -------------------------- Controls -------------------------- #
fit_checkbox = st.checkbox(
    "Apply Rabi Fit", value=st.session_state.get("fit_checkbox", False)
)
st.session_state.fit_checkbox = fit_checkbox

cool_checkbox = st.checkbox(
    "Apply cool reset", value=st.session_state.get("apply_cool", True)
)
st.session_state.config["apply_cool"] = cool_checkbox

# -------------------------- Run Experiment -------------------------- #
if st.button("Run"):
    st.session_state.timerabi = TimeRabi(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.timerabi.run(py_avg=pyavg, fit=st.session_state.fit_checkbox)
    st.success("Experiment completed!")

# -------------------------- Plot Result -------------------------- #
if (
    st.session_state.timerabi
    and hasattr(st.session_state.timerabi, "iqdata")
    and st.session_state.timerabi.iqdata is not None
):
    length = st.session_state.timerabi.length
    iqdata = st.session_state.timerabi.iqdata

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(length, np.abs(iqdata), label="Magnitude", marker="o", markersize=5)

    if st.session_state.fit_checkbox:
        pOpt, _ = fitter.fitdecaysin(length, np.abs(iqdata))
        pi, pi2 = pipulse_analyze(pOpt)
        ax.plot(length, fitter.decaysin(length, *pOpt), label="Fit")
        ax.axvline(pi, ls="--", c="red", label=f"$\\pi$ = {pi:.2f} us")
        ax.axvline(pi2, ls="--", c="green", label=f"$\\pi/2$ = {pi2:.2f} us")

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("ADC unit (a.u)")
    ax.set_title("Time Rabi ge")
    ax.legend()
    st.pyplot(fig)
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")

# -------------------------- Notes & Save -------------------------- #
st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.timerabi.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("SaveLabber"):
        st.session_state.timerabi.save_labber()
        st.success("LabberData saved successfully!")
