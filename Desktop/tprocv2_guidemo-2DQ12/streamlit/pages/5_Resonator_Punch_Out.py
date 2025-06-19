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
)
from single_qubit_pyscrip.SQ002b_res_punchout_ge import (
    SingleToneSpectroscopyPunchoutProgram,
)
from qick.asm_v2 import QickSweep1D

st.set_page_config(layout="wide")
st.title("Resonator SingleTone Spectroscopy Punchout")

# Initialize experiment metadata
st.session_state.expt_name = "002b_res_punchout_ge"
Qubit = "Q" + str(st.session_state.QubitIndex)

# Merge configurations into single dictionary
st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

# Ensure state keys exist
for key in ["punchout", "config", "punchout_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class SingleToneSpectroscopyPunchout:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.freqs = None
        self.gains = None
        self.soc = st.session_state.soc

    def run(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

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

            data = np.abs(self.iqdata)
            data_norm = np.array(
                [
                    (row - np.min(row)) / (np.max(row) - np.min(row))
                    if np.max(row) != np.min(row)
                    else row
                    for row in data
                ]
            )

            ax.cla()
            im = ax.pcolorfast(self.freqs, self.gains, data_norm)
            ax.set_title(f"Average: {i + 1} / {py_avg}")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("DAC Gain")
            ax.grid(False)
            placeholder.pyplot(fig)
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (py_avg - (i + 1))
            status_text.markdown(
                f"**Estimated time remaining: {format_time(remaining)}**"
            )
            progress_bar.progress((i + 1) / py_avg)
        plt.close(fig)
        placeholder.empty()

        if self.iq_list is not None:
            fig_final, ax_final = plt.subplots(figsize=(8, 4))
            ax_final.set_title("Resonator ge Punchout")
            ax_final.set_xlabel("Frequency (MHz)")
            ax_final.set_ylabel("DAC Gain (a.u)")
            im = ax_final.pcolorfast(self.freqs, self.gains, data_norm)
            fig_final.colorbar(im, ax=ax_final, label="Normalized Amplitude")
            st.session_state.punchout_fig = fig_final
            st.session_state.timetag = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            plt.close(fig_final)

    def save(self):
        file_path = get_next_filename(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
            suffix=".h5",
        )
        st.write(
            f"Experiment name: {st.session_state.expt_name}_Q{st.session_state.QubitIndex}"
        )
        st.write(f"Current data file: {file_path}")

        data_dict = {
            "experiment_name": "res_punch_out",
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,
            "y_name": "DAC Gain (a.u)",
            "y_value": self.gains,
            "z_name": "iq_list",
            "z_value": self.iqdata,
        }
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveh5(file_path, data_dict, result=result_dict)

    def save_labber(self):
        if self.freqs is None or self.iq_list is None:
            st.error("No data available. Run the experiment first.")
            return

        file_path = get_next_filename_labber(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
        )
        st.write(f"Current data file: {file_path}")

        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e3},
            y_info={"name": "Gain", "unit": "a.u", "values": self.gains},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": self.iqdata,
            },
            comment=result_dict["notes"],
            tag="OneTone",
        )


# UI Input Controls
col1, col2, col3 = st.columns(3)
with col1:
    start_freq = st.number_input(
        "Start Frequency (MHz)", min_value=0, value=4000, step=1
    )
with col2:
    stop_freq = st.number_input(
        "Stop Frequency (MHz)", min_value=start_freq, value=9000, step=1
    )
with col3:
    freq_steps = st.number_input(
        "Steps:", min_value=1, max_value=1000, value=101, step=1
    )

col1, col2, col3 = st.columns(3)
with col1:
    start_gain = st.number_input(
        "Start Gain (a.u)", min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )
with col2:
    stop_gain = st.number_input(
        "Stop Gain (a.u)", min_value=start_gain, max_value=1.0, value=0.5, step=0.01
    )
with col3:
    gain_steps = st.number_input(
        "Gain Steps:", min_value=1, max_value=100, value=5, step=1
    )

py_avg = st.number_input(
    "Soft average #:", min_value=1, max_value=10000, value=10, step=1
)

# Update configuration for sweeps
st.session_state.config.update(
    {
        "f_steps": freq_steps,
        "res_freq_ge": QickSweep1D("freqloop", start_freq, stop_freq),
        "g_steps": gain_steps,
        "res_gain_ge": QickSweep1D("gainloop", start_gain, stop_gain),
        "py_avg": py_avg,
    }
)

#### Sidebar Config View & Parameter Editor ####
st.sidebar.title("Experiment Configuration")
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


# Run experiment
if st.button("Run"):
    st.session_state.punchout = SingleToneSpectroscopyPunchout(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.punchout.run(py_avg=py_avg)
    st.success("Experiment completed!")

# Plot result and save options
if st.session_state.punchout and st.session_state.punchout_fig:
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")
    st.pyplot(st.session_state.punchout_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results..."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            st.session_state.punchout.save()
            st.success("Data saved successfully!")
    with col2:
        if st.button("SaveLabber"):
            st.session_state.punchout.save_labber()
            st.success("LabberData saved successfully!")
