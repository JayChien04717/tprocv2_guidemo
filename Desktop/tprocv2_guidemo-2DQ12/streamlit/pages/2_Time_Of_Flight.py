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
from single_qubit_pyscrip.SQ001_time_of_flight import LoopbackProgram

st.title("Time Of Flight Experiment")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "001_tof"
Qubit = "Q" + str(st.session_state.QubitIndex)

# Merge all configurations into one dictionary
st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

# Ensure session state variables exist
for key in ["tof", "config", "tof_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class TOF:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.t = None

    def run(self, pyavg):
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        iq_sum = 0
        self.iq_list = None

        fig, ax = plt.subplots()
        placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = datetime.datetime.now()

        for i in range(pyavg):
            self.iq_list = prog.acquire_decimated(st.session_state.soc, soft_avgs=1)
            if i == 0:
                self.t = prog.get_time_axis(ro_index=0)
            iq_data = self.iq_list[0].dot([1, 1j])
            iq_sum = iq_data if i == 0 else iq_sum + iq_data
            avg_iq = iq_sum / (i + 1)

            # liveplot
            ax.cla()
            ax.plot(self.t, avg_iq.real, label="I")
            ax.plot(self.t, avg_iq.imag, label="Q")
            ax.plot(self.t, np.abs(avg_iq), label="Magnitude")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title(f"Time Of Flight (Average: {i + 1} / {pyavg})")
            ax.legend()
            placeholder.pyplot(fig)

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (pyavg - (i + 1))
            status_text.markdown(
                f"**Estimated time remaining: {format_time(remaining)}**"
            )
            progress_bar.progress((i + 1) / pyavg)

        plt.close(fig)
        placeholder.empty()  # 關掉最後一個佔位

        # 存圖，畫最終平均
        if self.iq_list is not None:
            fig_final, ax_final = plt.subplots()
            ax_final.plot(self.t, avg_iq.real, label="I")
            ax_final.plot(self.t, avg_iq.imag, label="Q")
            ax_final.plot(self.t, np.abs(avg_iq), label="Magnitude")
            ax_final.set_xlabel("Time (us)")
            ax_final.set_ylabel("ADC unit (a.u)")
            ax_final.set_title("Time Of Flight (Final)")
            ax_final.legend()
            st.session_state.tof_fig = fig_final
            st.session_state.timetag = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            plt.close(fig_final)

    def save(self):
        data_path = st.session_state.datafile
        exp_name = st.session_state.expt_name + "_Q" + str(st.session_state.QubitIndex)
        st.write(f"Experiment name: {exp_name}")
        file_path = get_next_filename(data_path, exp_name, suffix=".h5")
        st.write(f"Current data file: {file_path}")

        data_dict = {
            "x_name": "Time(us)",
            "x_value": self.t,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0].dot([1, 1j]),
        }

        saveh5(file_path, data_dict)
        st.success("Data saved successfully!")

    def save_labber(self):
        """
        Save experimental data into an HDF5 file.
        """
        if self.t is None or self.iq_list is None:
            st.error("No data available. Run the experiment first.")
            return

        data_path = st.session_state.datafile
        exp_name = f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}"
        st.write(f"Experiment name: {exp_name}")

        file_path = get_next_filename_labber(data_path, exp_name)
        st.write(f"Current data file: {file_path}")

        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "Time", "unit": "s", "values": self.t * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": self.iq_list[0].dot([1, 1j]),
            },
            comment=f"{result_dict['notes']}",
            tag="OneTone",
        )


# **Soft Average Configuration**
col_spacer, col_right = st.columns([2, 1])
with col_spacer:
    pyavg = st.number_input(
        "Soft average #", min_value=1, max_value=10000, value=500, step=1
    )


# --- UI Input with Sync ---
with col_right:
    st.markdown("### ")

    res_gain = st.number_input(
        "Resonator gain", min_value=1e-4, max_value=1.0, value=1.0, step=1e-4
    )
    sync_param_to_config("res_gain_ge", res_gain, target_cfg_group="readout_cfg")

    res_length = st.number_input(
        "Resonator Pulse Length (us)",
        min_value=0.01,
        max_value=10.0,
        value=0.5,
        step=0.1,
    )
    sync_param_to_config("res_length", res_length, target_cfg_group="readout_cfg")

    ro_length = st.number_input(
        "Readout Length (us)", min_value=0.01, max_value=20.0, value=1.5, step=0.1
    )
    sync_param_to_config("ro_length", ro_length, target_cfg_group="readout_cfg")

####################################
# ---- Sidebar Configurations ---- #
####################################

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


################################
# ---- Streamlet function ---- #
################################

# Ensure session state variables exist
for key in ["tof", "config", "tof_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---- Run Experiment ---- #
if st.button("Run"):
    st.session_state.tof = TOF(st.session_state.soccfg, st.session_state.config)
    st.session_state.tof.run(pyavg)
    st.success("Experiment completed!")

if st.session_state.get("tof_fig", None):
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")
    st.pyplot(st.session_state.tof_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results..."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            st.session_state.tof.save()
            st.success("Data saved successfully!")
    with col2:
        if st.button("SaveLabber"):
            st.session_state.tof.save_labber()
            st.success("LabberData saved successfully!")
