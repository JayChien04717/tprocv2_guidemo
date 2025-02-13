import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ008_T1_ge import T1Program
from qick.asm_v2 import QickSpan, QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.title("T1 ge")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "008_T1_ge"
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
for key in ["t1", "config", "t1_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class T1ge:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.t = None

    def run(self, reps):
        prog = T1Program(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']
        st_progress = st.progress(0)
        self.iq_list = prog.acquire(
            st.session_state.soc, soft_avgs=py_avg, st_progress=st_progress)
        self.t = prog.get_time_param('wait', "t", as_array=True)

    def plot(self, fit=False):
        plt.rcParams.update({'font.size': 14})
        if self.iq_list is not None:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(self.t, np.abs(
                self.iq_list[0][0].dot([1, 1j])), label="Magnitude",  marker='o', markersize=5)

            if fit:
                pOpt, pCov = fitter.fitexp(
                    self.t, np.abs(self.iq_list[0][0].dot([1, 1j])))
                ax.plot(self.t, fitter.expfunc(
                    self.t, *pOpt), label=f"Fit T1 = {pOpt[3]:.2f} us")

            ax.legend(loc="upper right")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title("T1 ge")
            # Save to session state
            st.session_state.t1_fig = fig
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
            "experiment_name": "t1_ge",
            "x_name": "Time (us)",
            "x_value": self.t,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }
        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}

        saveh5(file_path, data_dict, result=result_dict)


col1, col2, col3 = st.columns(3)

with col1:
    start_t = st.number_input(
        "Start Time (us)", min_value=0.0, value=0.1, step=0.1)

with col2:
    stop_t = st.number_input(
        "Stop Time (us)", min_value=start_t, value=1.0, step=0.1)

with col3:
    steps = st.number_input("Steps:", min_value=1,
                            max_value=1000, value=101, step=1)

st.session_state.config.update(
    [('steps', steps), ('wait_time', QickSweep1D('waitloop', start_t, stop_t))])

# **Soft Average Configuration**

pyavg = st.number_input("Soft average #:", min_value=1,
                        max_value=10000, value=10, step=1)
relax_delay = st.number_input("relaxatoin time (us):", min_value=1,
                              max_value=1000, value=10, step=1)

st.session_state.config['relax_delay'] = relax_delay
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
    st.session_state.t1 = T1ge(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.t1.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.t1.plot(fit=st.session_state.fit_checkbox)

if "t1" in st.session_state and st.session_state.t1:

    st.session_state.t1.plot(fit=st.session_state.fit_checkbox)

    if st.session_state.t1_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.t1_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results...")
    if st.button("Save"):
        st.session_state.t1.save()
        st.success("Data saved successfully!")
