import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
from single_qubit_pyscrip.SQ002b_res_punchout_ge import SingleToneSpectroscopyPunchoutProgram
from qick.asm_v2 import QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.set_page_config(layout="wide")  # Enable wide mode for better layout
st.title("Resonator SingleTone Spectroscopy Punchout")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "002b_res_punchout_ge"
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
for key in ["punchout", "config", "punchout_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class SingleToneSpectroscopyPunchout:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg

    def run(self, reps):
        prog = SingleToneSpectroscopyPunchoutProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']
        self.iq_list = prog.acquire(
            st.session_state.soc, soft_avgs=py_avg, progress=True)
        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)
        self.gains = prog.get_pulse_param("res_pulse", "gain", as_array=True)

    def plot(self):
        avg_abs, avg_angle = (np.abs(self.iq_list[0][0].dot([1, 1j])),
                              np.angle(self.iq_list[0][0].dot([1, 1j])))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        plt.pcolormesh(self.freqs, self.gains, np.abs(
            self.iq_list[0][0].dot([1, 1j])),  shading="Auto")
        for i, d in enumerate([avg_abs, avg_angle]):
            if i == 0:
                pcm = axes[i].pcolormesh(
                    self.freqs, self.gains, d, shading="Auto")
            else:
                pcm = axes[i].pcolormesh(
                    self.freqs, self.gains, np.unwrap(d), shading="Auto", cmap="bwr")
            axes[i].set_ylabel("Gain")
            axes[i].set_xlabel("Freq(MHz)")
            axes[i].set_title("Amp" if i == 0 else "IQ phase (rad)")
            plt.colorbar(pcm, ax=axes[i])

        st.session_state.punchout_fig = fig
        st.session_state.timetag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self):
        data_path = st.session_state.datafile
        exp_name = st.session_state.expt_name + \
            '_Q' + str(st.session_state.QubitIndex)
        st.write(f'Experiment name: {exp_name}')
        file_path = get_next_filename(data_path, exp_name, suffix='.h5')
        st.write(f'Current data file: {file_path}')

        data_dict = {
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,
            "y_name": "DAC Gain (a.u)",
            "y_value": self.gains,
            "z_name": "iq_list",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }

        result = {
            "T1": "350us",
            "T2": "130us"
        }
        saveh5(file_path, data_dict, result)


# Define the layout with two columns
# col1 (wider) for main experiment, col2 for sidebar config
col1, col2 = st.columns([2, 1])
with col1:
    # User inputs for frequency and gain
    start_freq = st.number_input(
        "Start Frequency (MHz)", min_value=0, value=4000, step=1)
    stop_freq = st.number_input(
        "Stop Frequency (MHz)", min_value=start_freq, value=5000, step=1)
    steps_freq = st.number_input(
        "Steps:", min_value=1, max_value=10000, value=101, step=1)

    start_gain = st.number_input(
        "Start Gain (a.u)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    stop_gain = st.number_input(
        "Stop Gain (a.u)", min_value=start_gain, max_value=1.0, value=0.5, step=0.01)
    steps_gain = st.number_input(
        "Gain Steps:", min_value=1, max_value=100, value=5, step=1)

    py_avg = st.number_input(
        "Soft average #:", min_value=1, max_value=1000, value=10, step=1)
    st.session_state.config.update({
        'f_steps': steps_freq,
        'res_freq_ge': QickSweep1D('freqloop', start_freq, stop_freq),
        'g_steps': steps_gain,
        'res_gain_ge': QickSweep1D('gainloop', start_gain, stop_gain)
    })

    st.session_state.config['py_avg'] = py_avg

# Buttons to Run and Save
if st.button("Run"):
    st.session_state.onetonepunchout = SingleToneSpectroscopyPunchout(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.onetonepunchout.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.onetonepunchout.plot()

if st.session_state.onetonepunchout:
    if st.button("Save"):
        st.session_state.onetonepunchout.save()
        st.success("Data saved successfully!")

# Placeholder for displaying plots
if "punchout_fig" in st.session_state and st.session_state.punchout_fig:
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")
    st.pyplot(st.session_state.punchout_fig)

# Sidebar for configuration settings
with col2:
    st.sidebar.title("Experiment Configuration")

    with st.sidebar.expander("🖥 Hardware Config"):
        st.json(st.session_state.hw_cfg)

    with st.sidebar.expander("📡 Readout Config"):
        st.json(st.session_state.readout_cfg)

    with st.sidebar.expander("⚛️ Qubit Config"):
        st.json(st.session_state.qubit_cfg)

    with st.sidebar.expander("🔬 Experiment Config"):
        st.json(st.session_state.expt_cfg)

    # Dropdown to update parameters dynamically
    config_key = st.sidebar.selectbox(
        "Select Config Parameter to Update", list(st.session_state.config.keys()))

    new_value = st.sidebar.text_input(f"New value for {config_key}:")

    if st.sidebar.button("Update Config"):
        st.session_state.config[config_key] = eval(
            new_value) if new_value.isnumeric() else new_value
        st.rerun()
