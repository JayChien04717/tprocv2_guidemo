from single_qubit_pyscrip.system_tool import select_config_idx
import streamlit as st
from single_qubit_pyscrip.SQ001_time_of_flight import LoopbackProgram
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.title("Time Of Flight Experiment")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "001_tof"
Qubit = 'Q' + str(st.session_state.QubitIndex)

# Merge all configurations into one dictionary
st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex
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

    def run(self):
        prog = LoopbackProgram(
            self.soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        self.iq_list = prog.acquire_decimated(
            st.session_state.soc, soft_avgs=self.cfg['py_avg'])
        self.t = prog.get_time_axis(ro_index=0)

    def plot(self):
        plt.rcParams.update({'font.size': 14})
        if self.iq_list is not None:
            fig, ax = plt.subplots()
            ax.plot(self.t, self.iq_list[0].T[0], label="I")
            ax.plot(self.t, self.iq_list[0].T[1], label="Q")
            ax.plot(self.t, np.abs(
                (self.iq_list[0]).dot([1, 1j])), label="Magnitude")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title("Time Of Flight")
            ax.legend()

            # Save to session state
            st.session_state.tof_fig = fig
            st.session_state.timetag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.warning("No data to plot. Please run the experiment first.")

    def save(self):
        data_path = st.session_state.datafile
        exp_name = st.session_state.expt_name + \
            '_Q' + str(st.session_state.QubitIndex)
        st.write(f'Experiment name: {exp_name}')
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        st.write(f'Current data file: {file_path}')

        data_dict = {
            "x_name": "Time(us)",
            "x_value": self.t,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0].dot([1, 1j])
        }

        saveh5(file_path, data_dict)
        st.success("Data saved successfully!")


# **Soft Average Configuration**
pyavg = st.number_input("Soft average #:", min_value=1,
                        max_value=1000, value=10, step=1)
st.session_state.config['py_avg'] = pyavg

####################################
# ---- Sidebar Configurations ---- #
####################################
st.sidebar.title("Experiment Configuration")

# Ensure QubitIndex is stored in session state
qubit_index = int(st.session_state.get(
    "QubitIndex", 1))  # Default to 1 if not set

# Display indexed configurations in sidebar
with st.sidebar.expander("🖥 Hardware Config"):
    st.json(st.session_state.hw_cfg)

with st.sidebar.expander("📡 Readout Config"):
    st.json(select_config_idx(st.session_state.readout_cfg, idx=qubit_index))

with st.sidebar.expander("⚛️ Qubit Config"):
    st.json(select_config_idx(st.session_state.qubit_cfg, idx=qubit_index))

with st.sidebar.expander("🔬 Experiment Config"):
    st.json(st.session_state.expt_cfg)

# Dropdown to update individual configuration values dynamically
config_key = st.sidebar.selectbox(
    "Select Config Parameter to Update", list(st.session_state.config.keys()))

new_value = st.sidebar.text_input(f"New value for {config_key}:")
if st.sidebar.button("Update Config"):
    # Update merged dictionary
    st.session_state.config[config_key] = eval(
        new_value) if new_value.isnumeric() else new_value

    # Split back into individual configs
    st.session_state.hw_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.hw_cfg}
    st.session_state.readout_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.readout_cfg}
    st.session_state.qubit_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.qubit_cfg}
    st.session_state.expt_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.expt_cfg}

    # Force rerun to reflect updates
    st.rerun()


################################
# ---- Streamlet function ---- #
################################

# Ensure session state variables exist
for key in ["tof", "config", "tof_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---- Run Experiment ---- #
if st.button("Run"):
    st.session_state.tof = TOF(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.tof.run()
    st.success("Experiment completed!")
    st.session_state.tof.plot()

if st.session_state.tof:
    if st.button("Save"):
        st.session_state.tof.save()

# ---- Display Graph ---- #
if st.session_state.tof_fig:
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")
    st.pyplot(st.session_state.tof_fig)
