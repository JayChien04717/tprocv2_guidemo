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
from single_qubit_pyscrip.SQ002_res_spec_ge import SingleToneSpectroscopyProgram
from single_qubit_pyscrip.abcd_rf_fit import analyze
from qick.asm_v2 import QickSweep1D

# ---------- UI Header ---------- #
st.title("Resonator OneTone Spectroscopy")
st.session_state.expt_name = "002_onetone"
Qubit = "Q" + str(st.session_state.QubitIndex)

st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

for key in ["onetone", "config"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------- Measurement Class ---------- #
class ResonatorOnetone:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.freqs = None
        self.soc = st.session_state.soc

    def run(self, py_avg, fit=False):
        iq = 0
        prog = SingleToneSpectroscopyProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

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

        plt.close(fig)
        placeholder.empty()
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
            "experiment_name": "res_spec_ge",
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iqdata,
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
            x_info={"name": "Frequency", "unit": "Hz", "values": self.freqs * 1e9},
            z_info={"name": "Signal", "unit": "a.u.", "values": self.iqdata},
            comment=result_dict["notes"],
            tag="OneTone",
        )


# ---------- Config Inputs ---------- #
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
    steps = st.number_input("Steps", min_value=1, max_value=1000, value=101, step=1)

st.session_state.config.update(
    {"steps": steps, "res_freq_ge": QickSweep1D("freqloop", start_freq, stop_freq)}
)

col_spacer, col_right = st.columns([2, 1])
with col_spacer:
    pyavg = st.number_input(
        "Soft average #", min_value=1, max_value=10000, value=10, step=1
    )
    res_gain = st.number_input(
        "Resonator gain", min_value=1e-4, max_value=1.0, value=0.5, step=1e-4
    )
st.session_state.config.update({"res_gain_ge": res_gain})

with col_right:
    st.header("Optional Parameters")

    res_length = st.number_input(
        "Resonator Pulse Length (us)",
        min_value=0.01,
        max_value=10.0,
        value=5.0,
        step=0.1,
    )
    sync_param_to_config("res_length", res_length, target_cfg_group="readout_cfg")
    ro_length = st.number_input(
        "Readout Length (us)", min_value=0.01, max_value=20.0, value=5.0, step=0.1
    )
    sync_param_to_config("ro_length", ro_length, target_cfg_group="readout_cfg")

# ---------- Sidebar Config ---------- #
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

# ---------- Measurement Execution ---------- #
fit_checkbox = st.checkbox(
    "Fit Data", value=st.session_state.get("fit_checkbox", False)
)
st.session_state.fit_checkbox = fit_checkbox

circle_fit_checkbox = st.checkbox(
    "Do Circle Fit", value=st.session_state.get("circle_fit_checkbox", False)
)
st.session_state.circle_fit_checkbox = circle_fit_checkbox

if st.button("Run"):
    st.session_state.onetone = ResonatorOnetone(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.onetone.run(py_avg=pyavg, fit=st.session_state.fit_checkbox)
    st.success("Experiment completed!")

# ---------- Plot Result ---------- #
if (
    st.session_state.onetone
    and hasattr(st.session_state.onetone, "iqdata")
    and st.session_state.onetone.iqdata is not None
):
    freqs = st.session_state.onetone.freqs
    iqdata = st.session_state.onetone.iqdata

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(iqdata), label="Magnitude", marker="o", markersize=5)

    if st.session_state.fit_checkbox:
        pOpt, _ = fitter.fit_asym_lor(freqs, np.abs(iqdata))
        f0 = pOpt[2]
        st.session_state.fitted_f0 = f0
        ax.plot(
            freqs,
            fitter.asym_lorfunc(freqs, *pOpt),
            label=f"Fit res freq = {f0:.4f} MHz",
        )

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("ADC unit (a.u)")
    ax.set_title("Resonator OneTone Spectroscopy")
    ax.legend()
    st.pyplot(fig)
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")

# ---------- Circle Fit ---------- #
if (
    st.session_state.circle_fit_checkbox
    and st.session_state.onetone
    and st.session_state.onetone.iqdata is not None
):
    x = st.session_state.onetone.freqs
    y = st.session_state.onetone.iqdata
    fit = analyze(x * 1e6, y, "hm", fit_edelay=True)

    st.write("### Circle Fit Result")
    fig_circle = plt.figure(figsize=(6, 6))
    fit.plot(fig=fig_circle)
    st.pyplot(fig_circle)

    param = fit.tolist()
    result_dict = {
        "Fres (MHz)": f"{param[0] / 1e6:.4f}",
        "Qi": f"{int(abs(param[0] / (param[1] - param[2]))):,}",
        "absQc": f"{int(abs(param[0] / param[2])):,}",
        "Ql": f"{int(param[0] / param[1]):,}",
        "κ (MHz)": f"{param[1] * 1e-6:.3f}",
    }
    st.write("#### Extracted Parameters")
    st.table(result_dict)

    if st.button("Update res_freq_ge"):
        sync_param_to_config(
            "res_freq_ge", round(param[0] / 1e6, 4), target_cfg_group="readout_cfg"
        )

# ---------- Notes & Save ---------- #
st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.onetone.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("SaveLabber"):
        st.session_state.onetone.save_labber()
        st.success("LabberData saved successfully!")
