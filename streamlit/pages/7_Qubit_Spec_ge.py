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
from single_qubit_pyscrip.SQ003_qubit_spec_ge import PulseProbeSpectroscopyProgram
from qick.asm_v2 import QickSweep1D

# -------------------------- Initialization -------------------------- #
st.title("Qubit ge Spectroscopy")
st.session_state.expt_name = "003_qubit_spec_ge"
Qubit = "Q" + str(st.session_state.QubitIndex)

st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.cooling_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

for key in ["twotone", "config"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------------- Experiment Class -------------------------- #
class QubitTwotone:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.soc = st.session_state.soc
        self.freqs = None
        self.iq_list = None
        self.iqdata = None

    def run(self, py_avg, fit=False):
        iq = 0
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)
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
            ax.plot(self.freqs, np.abs(self.iqdata), **marker_style, label="Magnitude")
            ax.set_title(f"Average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("ADC units")

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
            "experiment_name": "qubit_spec_ge",
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j]),
        }
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveh5(path, data_dict, result=result_dict)

    def save_labber(self):
        if self.freqs is None or self.iq_list is None:
            st.error("No data available. Run the experiment first.")
            return
        path = get_next_filename_labber(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
        )
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e6},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": self.iq_list[0][0].dot([1, 1j]),
            },
            comment=result_dict["notes"],
            tag="TwoTone",
        )


# -------------------------- Parameters -------------------------- #
col1, col2, col3 = st.columns(3)
with col1:
    start_freq = st.number_input("Start Frequency (MHz)", min_value=1, value=1, step=1)
with col2:
    stop_freq = st.number_input(
        "Stop Frequency (MHz)", min_value=start_freq, value=6000, step=1
    )
with col3:
    steps = st.number_input("Steps", min_value=1, max_value=10000, value=101, step=1)

st.session_state.config.update(
    {
        "steps": steps,
        "qubit_freq_ge": QickSweep1D("freqloop", start_freq, stop_freq),
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
    qubit_mixer = st.number_input(
        "Qubit mixer freq", min_value=10.0, max_value=1e10, value=3000.0, step=1.0
    )
    sync_param_to_config("qubit_mixer_freq", qubit_mixer, target_cfg_group="qubit_cfg")

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
    "Apply Lorentzian Fit", value=st.session_state.get("fit_checkbox", False)
)
st.session_state.fit_checkbox = fit_checkbox

cool_checkbox = st.checkbox(
    "Apply cool reset", value=st.session_state.get("apply_cool", True)
)
st.session_state.config["apply_cool"] = cool_checkbox


if st.button("Run"):
    st.session_state.twotone = QubitTwotone(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.twotone.run(py_avg=pyavg, fit=st.session_state.fit_checkbox)
    st.success("Experiment completed!")

# -------------------------- Plot Result -------------------------- #
if (
    st.session_state.twotone
    and hasattr(st.session_state.twotone, "iqdata")
    and st.session_state.twotone.iqdata is not None
):
    freqs = st.session_state.twotone.freqs
    iqdata = st.session_state.twotone.iqdata

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(iqdata), label="Magnitude", marker="o", markersize=5)

    if st.session_state.fit_checkbox:
        pOpt, _ = fitter.fitlor(freqs, np.abs(iqdata))
        f0 = pOpt[2]
        st.session_state.fitted_f0 = f0
        ax.plot(
            freqs, fitter.lorfunc(freqs, *pOpt), label=f"Fit f_ge freq = {f0:.4f} MHz"
        )

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("ADC unit (a.u)")
    ax.set_title("Qubit ge Spectroscopy")
    ax.legend()
    st.pyplot(fig)
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")

# -------------------------- Update Config -------------------------- #
if st.button("Update qubit_freq_ge"):
    if st.session_state.get("fitted_f0") is not None:
        f0 = st.session_state.fitted_f0
        sync_param_to_config(
            "qubit_freq_ge", round(f0, 4), target_cfg_group="qubit_cfg"
        )
        sync_param_to_config(
            "qubit_mixer_freq", round(f0, 4), target_cfg_group="qubit_cfg"
        )
        st.success(f"Updated qubit config with f_ge = {f0:.4f} MHz")
    else:
        st.warning("Please enable fit checkbox and run fitting before updating config.")

# -------------------------- Notes & Save -------------------------- #
st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.twotone.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("SaveLabber"):
        st.session_state.twotone.save_labber()
        st.success("LabberData saved successfully!")
