import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ003_qubit_spec_ge import PulseProbeSpectroscopyProgram
from qick.asm_v2 import QickSpan, QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.title("Qubit ge Spectroscopy")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "003_qubit_spec_ge"
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
for key in ["twotone", "config", "twotone_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class QubitTwotone:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.freqs = None

    def run(self, reps):
        prog = PulseProbeSpectroscopyProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']
        st_progress = st.progress(0)
        self.iq_list = prog.acquire(
            st.session_state.soc, soft_avgs=py_avg, st_progress=st_progress)
        self.freqs = prog.get_pulse_param("qubit_pulse", "freq", as_array=True)

    def plot(self, fit=False):
        plt.rcParams.update({'font.size': 14})
        if self.iq_list is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(self.freqs, np.abs(
                self.iq_list[0][0].dot([1, 1j])), label="Magnitude", marker='o', markersize=5)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title("Qubit ge Spectroscopy")
            ax.legend()
            if fit:
                pOpt, pCov = fitter.fitlor(
                    self.freqs, np.abs(self.iq_list[0][0].dot([1, 1j])))
                resonacefreq = pOpt[2]
                ax.plot(self.freqs, fitter.lorfunc(
                    self.freqs, *pOpt), label=f"Fit res freq = {resonacefreq:.0f}MHz")
                ax.legend()
            # Save to session state
            st.session_state.twotone_fig = fig
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
            "experiment_name": "qubit_spec_ge",
            "x_name": "Frequency (MHz)",
            "x_value": self.freqs,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }
        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}

        saveh5(file_path, data_dict, result=result_dict)


col1, col2, col3 = st.columns(3)
with col1:
    start_freq = st.number_input(
        "Start Frequency (MHz)", min_value=0, value=4000, step=1)

with col2:
    stop_freq = st.number_input(
        "Stop Frequency (MHz)", min_value=start_freq, value=5000, step=1)

with col3:
    steps = st.number_input("Steps:", min_value=1,
                            max_value=1000, value=101, step=1)

st.session_state.config.update(
    [('steps', steps), ('qubit_freq_ge', QickSweep1D('freqloop', start_freq, stop_freq))])

# **Soft Average Configuration**
pyavg = st.number_input("Soft average #:", min_value=1,
                        max_value=10000, value=10, step=1)
st.session_state.config['py_avg'] = pyavg

######################################
# ---- Sidebar Configurations ---- #
######################################
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


###############################
# ---- Streamlit Functions ---- #
###############################

fit_checkbox = st.checkbox(
    "Fit Data", value=st.session_state.get("fit_checkbox", False))
st.session_state.fit_checkbox = fit_checkbox


if st.button("Run"):
    st.session_state.twotone = QubitTwotone(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.twotone.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.twotone.plot(fit=st.session_state.fit_checkbox)

if "twotone" in st.session_state and st.session_state.twotone:

    st.session_state.twotone.plot(fit=st.session_state.fit_checkbox)

    if st.session_state.twotone_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.twotone_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results...")
    if st.button("Save"):
        st.session_state.twotone.save()
        st.success("Data saved successfully!")
