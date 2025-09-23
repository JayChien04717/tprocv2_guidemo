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
from single_qubit_pyscrip.sidebar_config import (
    show_sidebar_config,
    config_update_sidebar,
    sync_param_to_config,
)
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ006_Ramsey_ge import RamseyProgram
from qick.asm_v2 import QickSweep1D

# -------------------------- Initialization -------------------------- #
st.title("Ramsey ge")
st.session_state.expt_name = "006_Ramsey_ge"
Qubit = "Q" + str(st.session_state.QubitIndex)

st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.cooling_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

for key in ["ramsey", "ramsey_fig", "ramsey_fit_detune"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------------- Experiment Class (Liveplot) -------------------------- #
class Ramseyge:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.soc = st.session_state.soc
        self.iq_list = None
        self.t = None
        self.iqdata = None

    def run(self, py_avg, fit=False):
        prog = RamseyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.t = prog.get_time_param("wait", "t", as_array=True)
        iq = 0
        fig, ax = plt.subplots(figsize=(8, 4))
        marker_style = {"marker": "o", "markersize": 5, "alpha": 0.7, "linestyle": "-"}
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
            ax.plot(self.t, np.abs(self.iqdata), **marker_style, label="Magnitude")
            ax.set_title(f"Average: {i + 1} / {py_avg}")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit (a.u)")

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
        st.session_state.timetag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self):
        path = get_next_filename(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
            suffix=".h5",
        )
        data_dict = {
            "experiment_name": "ramsey_ge",
            "x_name": "Time (us)",
            "x_value": self.t,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iqdata,
        }
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveh5(path, data_dict, result=result_dict)

    def save_labber(self):
        if self.t is None or self.iqdata is None:
            st.error("No data available. Run the experiment first.")
            return
        path = get_next_filename_labber(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
        )
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=path,
            x_info={"name": "Time", "unit": "s", "values": self.t},
            z_info={"name": "Signal", "unit": "a.u.", "values": self.iqdata},
            comment=result_dict["notes"],
            tag="T2",
        )


# -------------------------- Parameters -------------------------- #
col1, col2, col3 = st.columns(3)
with col1:
    start_t = st.number_input("Start Time (us)", min_value=0.0, value=0.0, step=0.1)
with col2:
    stop_t = st.number_input("Stop Time (us)", min_value=start_t, value=5.0, step=0.1)
with col3:
    steps = st.number_input("Steps", min_value=1, max_value=1000, value=101, step=1)

st.session_state.config.update(
    {
        "steps": steps,
        "wait_time": QickSweep1D("waitloop", start_t, stop_t),
    }
)
Ramsey_frequency = st.number_input(
    "Ramsey frequency (MHz):", min_value=0, max_value=10, value=2, step=1
)
pyavg = st.number_input(
    "Soft average #", min_value=1, max_value=10000, value=10, step=1
)
relax_delay = st.number_input(
    "Relaxation time (us):", min_value=1, max_value=1000, value=10, step=1
)
st.session_state.config["ramsey_freq"] = Ramsey_frequency
st.session_state.config["relax_delay"] = relax_delay
st.session_state.config["py_avg"] = pyavg

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

# -------------------------- Controls & Liveplot -------------------------- #
fit_checkbox = st.checkbox(
    "Fit Data", value=st.session_state.get("fit_checkbox", False)
)
st.session_state.fit_checkbox = fit_checkbox

cool_checkbox = st.checkbox(
    "Apply cool reset", value=st.session_state.get("apply_cool", True)
)
st.session_state.config["apply_cool"] = cool_checkbox

# -------------------------- Run Experiment -------------------------- #
if st.button("Run"):
    st.session_state.ramsey = Ramseyge(st.session_state.soccfg, st.session_state.config)
    st.session_state.ramsey.run(py_avg=pyavg, fit=st.session_state.fit_checkbox)
    st.success("Experiment completed!")

# -------------------------- Plot Result -------------------------- #
if (
    st.session_state.ramsey
    and hasattr(st.session_state.ramsey, "iqdata")
    and st.session_state.ramsey.iqdata is not None
):
    t = st.session_state.ramsey.t
    iqdata = st.session_state.ramsey.iqdata

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, np.abs(iqdata), label="Magnitude", marker="o", markersize=5)

    if st.session_state.fit_checkbox:
        pOpt, _ = fitter.fitdecaysin(t, np.abs(iqdata))
        ax.plot(t, fitter.decaysin(t, *pOpt), label=f"Fit Δ={pOpt[1]:.4f} MHz")
        st.session_state.ramsey_fit_detune = pOpt[1]
    else:
        st.session_state.ramsey_fit_detune = None
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("ADC unit (a.u)")
    ax.set_title("Ramsey ge")
    ax.legend()
    st.pyplot(fig)
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")

# -------------------------- Update Config -------------------------- #
if st.button("Correct Qubit detune"):
    detune = st.session_state.ramsey_fit_detune
    if detune is not None:
        if abs(detune - st.session_state.config["ramsey_freq"]) > 0.01:
            sync_param_to_config(
                "qubit_freq_ge",
                round(
                    st.session_state.config["qubit_freq_ge"]
                    - (float(detune - st.session_state.config["ramsey_freq"])),
                    4,
                ),
                target_cfg_group="qubit_cfg",
            )

            st.success(
                f"Over detune: ramsey_freq = {(detune - st.session_state.config['ramsey_freq']):.5f} MHz"
            )
        else:
            st.success("Detune < 0.01 MHz, no update needed.")

    else:
        st.warning("Please enable fit checkbox and run fitting before updating config.")

# -------------------------- Notes & Save -------------------------- #
st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.ramsey.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("SaveLabber"):
        st.session_state.ramsey.save_labber()
        st.success("LabberData saved successfully!")
