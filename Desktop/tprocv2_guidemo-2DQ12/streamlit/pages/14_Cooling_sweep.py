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
from single_qubit_pyscrip.SQ002d_cooling_ge import SingleToneSpectroscopyCoolingProgram
from qick.asm_v2 import QickSweep1D

st.set_page_config(layout="wide")
st.title("Cooling spec sweep")

# ============ Session State Initialization ============ #
st.session_state.expt_name = "002d_cooling_ge"
Qubit = "Q" + str(st.session_state.QubitIndex)

# 合併全部設定進一個 config dictionary
st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.cooling_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

# 確保所有狀態變數都已初始化
for key in ["cooling", "config", "cooling_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ============ 實驗 Class ============ #
class CoolingSweep:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.freqs1 = None
        self.freqs2 = None
        self.iqdata = None
        self.soc = st.session_state.soc

    def run(self, py_avg):
        iq = 0
        prog = SingleToneSpectroscopyCoolingProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )
        self.freqs1 = prog.get_pulse_param("cool_pulse1", "freq", as_array=True)
        self.freqs2 = prog.get_pulse_param("cool_pulse2", "freq", as_array=True)

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
            im = ax.pcolorfast(self.freqs1, self.freqs2, np.abs(self.iqdata))
            ax.set_title(f"Average: {i + 1} / {py_avg}")
            ax.set_xlabel(r"$|f,0\rangle - |e,0\rangle$")
            ax.set_ylabel(r"$|f,0\rangle - |g,1\rangle$")
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

        # Final result plot
        if self.iqdata is not None:
            fig_final, ax_final = plt.subplots(figsize=(12, 5))
            ax_final.set_title("Cooling Two-Tone Sweep")
            ax_final.set_xlabel(r"$|f,0\rangle - |e,0\rangle$")
            ax_final.set_ylabel(r"$|f,0\rangle - |g,1\rangle$")
            im = ax_final.pcolorfast(self.freqs1, self.freqs2, np.abs(self.iqdata))
            fig_final.colorbar(im, ax=ax_final, label="|Signal| (a.u.)")
            st.session_state.cooling_fig = fig_final
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
            "experiment_name": "cooling_spec_sweep",
            "x_name": "Freq1 (MHz)",
            "x_value": self.freqs1,
            "y_name": "Freq2 (MHz)",
            "y_value": self.freqs2,
            "z_name": "Signal",
            "z_value": self.iqdata,
        }
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveh5(file_path, data_dict, result=result_dict)

    def save_labber(self):
        if self.freqs1 is None or self.iqdata is None:
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
            x_info={"name": "Freq1", "unit": "Hz", "values": self.freqs1 * 1e6},
            y_info={"name": "Freq2", "unit": "Hz", "values": self.freqs2 * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": self.iqdata},
            comment=result_dict["notes"],
            tag="Cooling2D",
        )


# ============ UI Input Controls ============ #
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    start_freq_1 = st.number_input(
        "Start Freq1 (MHz)", min_value=10.0, value=1000.0, step=0.1
    )
with col2:
    stop_freq_1 = st.number_input(
        "Stop Freq1 (MHz)", min_value=start_freq_1, value=7000.0, step=0.1
    )
with col3:
    freq_steps_1 = st.number_input(
        "Steps 1:", min_value=1, max_value=1000, value=21, step=1
    )
with col4:
    ch1 = st.number_input("Cooling ch1:", min_value=1, max_value=10, value=5, step=1)
with col5:
    mixer1 = st.number_input(
        "Cooling ch1 mixer:", min_value=1, max_value=10000000000, value=3000, step=1
    )
with col6:
    gain1 = st.number_input(
        "Cooling ch1 Gain:", min_value=0.01, max_value=1.0, value=0.8, step=0.01
    )


col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    start_freq_2 = st.number_input(
        "Start Freq2 (MHz)", min_value=10.0, value=1000.0, step=0.1
    )
with col2:
    stop_freq_2 = st.number_input(
        "Stop Freq2 (MHz)", min_value=start_freq_2, value=7000.0, step=0.1
    )
with col3:
    freq_steps_2 = st.number_input(
        "Steps 2:", min_value=1, max_value=1000, value=21, step=1
    )
with col4:
    ch2 = st.number_input("Cooling ch2:", min_value=1, max_value=10, value=5, step=1)
with col5:
    mixer2 = st.number_input(
        "Cooling ch2 mixer:", min_value=1, max_value=10000000000, value=3000, step=1
    )
with col6:
    gain2 = st.number_input(
        "Cooling ch2 Gain:", min_value=0.01, max_value=1.0, value=0.8, step=0.01
    )


py_avg = st.number_input(
    "Soft average #:", min_value=1, max_value=10000, value=10, step=1
)

# Update config
st.session_state.config.update(
    {
        "f_steps1": freq_steps_1,
        "cool_freq_1": QickSweep1D("freqloop1", start_freq_1, stop_freq_1),
        "cool_ch1": ch1,
        "cool_mixer1": mixer1,
        "cool_gain_1": gain1,
        "f_steps2": freq_steps_2,
        "cool_freq_2": QickSweep1D("freqloop2", start_freq_2, stop_freq_2),
        "cool_ch2": ch2,
        "cool_mixer2": mixer2,
        "cool_gain_2": gain2,
        "py_avg": py_avg,
    }
)

####################################
# ---- Sidebar Configurations ---- #
####################################
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
####################################
# ---- Main Execution Controls ---- #
####################################
if st.button("Run"):
    st.session_state.cooling = CoolingSweep(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.cooling.run(py_avg=py_avg)
    st.success("Experiment completed!")

# ========== Cooling Point Tools ==========
st.markdown("### Cooling Point Tools")

# 勾選是否標示最大點
st.session_state.show_max_checkbox = st.checkbox(
    "Show max point on cooling plot", value=False
)

col1, col2 = st.columns(2)
with col1:
    if "max_cooling_point" in st.session_state:
        f1, f2 = st.session_state.max_cooling_point
        st.markdown(
            f"**Max Cooling Point:** Freq1 = {f1:.2f} MHz, Freq2 = {f2:.2f} MHz"
        )
with col2:
    if st.button("Update Cooling Parameter"):
        if "max_cooling_point" in st.session_state:
            f1, f2 = st.session_state.max_cooling_point
            sync_param_to_config(
                "cool_freq_1", round(f1, 4), target_cfg_group="cooling_cfg"
            )
            sync_param_to_config(
                "cool_freq_2", round(f2, 4), target_cfg_group="cooling_cfg"
            )
            st.success(
                f"Updated config with cooling point: f1={f1:.3f} MHz, f2={f2:.3f} MHz"
            )
        else:
            st.warning("Please run experiment and enable max point first.")

#########################################
# ---- Display Cooling Final Plot ---- #
#########################################
if (
    st.session_state.cooling
    and hasattr(st.session_state.cooling, "iqdata")
    and st.session_state.cooling.iqdata is not None
):
    fig_final, ax_final = plt.subplots(figsize=(12, 5))
    im = ax_final.pcolorfast(
        st.session_state.cooling.freqs1,
        st.session_state.cooling.freqs2,
        np.abs(st.session_state.cooling.iqdata),
    )
    fig_final.colorbar(im, ax=ax_final, label="|Signal| (a.u.)")
    ax_final.set_title("Cooling Two-Tone Sweep")
    ax_final.set_xlabel(r"$|f,0\rangle - |e,0\rangle$")
    ax_final.set_ylabel(r"$|f,0\rangle - |g,1\rangle$")

    if st.session_state.show_max_checkbox:
        iq = st.session_state.cooling.iqdata
        iq_metric = (np.abs(iq - np.mean(iq))) ** 2
        idx1, idx2 = np.unravel_index(np.argmax(iq_metric), iq_metric.shape)
        max1 = st.session_state.cooling.freqs1[idx2]
        max2 = st.session_state.cooling.freqs2[idx1]
        ax_final.plot(max1, max2, "rx", markersize=10, label="Max Point")
        ax_final.legend()
        st.session_state.max_cooling_point = (max1, max2)

    st.session_state.cooling_fig = fig_final
    st.session_state.timetag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.pyplot(st.session_state.cooling_fig)

#########################################
# ------------ Plot & Save ------------ #
#########################################
st.write(f"### Last Measurement Time: {st.session_state.timetag}")
st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.cooling.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("SaveLabber"):
        st.session_state.cooling.save_labber()
        st.success("LabberData saved successfully!")
