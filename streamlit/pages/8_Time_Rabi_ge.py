import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
from single_qubit_pyscrip.system_tool import hdf5_generator, get_next_filename_labber
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ004_time_rabi_ge import LengthRabiProgram
from qick.asm_v2 import QickSpan, QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime


st.title("Time Rabi ge")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "004_time_rabi_ge"
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
for key in ["timerabi", "config", "timerabi_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class TimeRabi:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.length = None

    def run(self, reps):
        prog = LengthRabiProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']
        st_progress = st.progress(0)
        self.iq_list = prog.acquire(
            st.session_state.soc, soft_avgs=py_avg, st_progress=st_progress)
        self.length = prog.get_pulse_param(
            "qubit_pulse", "length", as_array=True)

    def plot(self, fit=False):
        plt.rcParams.update({'font.size': 14})
        if self.iq_list is not None:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(self.length, np.abs(
                self.iq_list[0][0].dot([1, 1j])), label="Magnitude",  marker='o', markersize=5)

            if fit:
                pOpt, pCov = fitter.fitdecaysin(
                    self.length, np.abs(self.iq_list[0][0].dot([1, 1j])))

                if pOpt[2] > 180:
                    pOpt[2] = pOpt[2] - 360
                elif pOpt[2] < -180:
                    pOpt[2] = pOpt[2] + 360
                if pOpt[2] < 0:
                    pi_length = (1 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
                    pi2_length = (0 - pOpt[2] / 180) / 2 / pOpt[1]
                else:
                    pi_length = (3 / 2 - pOpt[2] / 180) / 2 / pOpt[1]
                    pi2_length = (1 - pOpt[2] / 180) / 2 / pOpt[1]
                ax.plot(self.length, fitter.decaysin(
                    self.length, *pOpt), label=f"Fit Rabi frequency = {pOpt[1]:.0f}MHz")
                ax.axvline(pi_length, ls="--", c="red",
                           label=f"$\pi$ length={pi_length:.2f} us")
                ax.axvline(pi2_length, ls="--", c="green",
                           label=f"$\pi2$ length={(pi2_length):.2f}us")
            ax.legend(loc="upper right")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title("Time Rabi ge")
            # Save to session state
            st.session_state.timerabi_fig = fig
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
            "experiment_name": "time_rabi_ge",
            "x_name": "Time (us)",
            "x_value": self.length,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }

        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}

        saveh5(file_path, data_dict, result=result_dict)

    def save_labber(self):
        """
        Save experimental data into an HDF5 file.
        """
        if self.length is None or self.iq_list is None:
            st.error("No data available. Run the experiment first.")
            return

        data_path = st.session_state.datafile
        exp_name = f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}"
        st.write(f'Experiment name: {exp_name}')

        file_path = get_next_filename_labber(data_path, exp_name)
        st.write(f'Current data file: {file_path}')

        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=file_path,
            x_info={'name': 'Time', 'unit': "s",
                    'values':  self.length*1e-6},
            z_info={'name': 'Signal', 'unit': 'a.u.',
                    'values': self.iq_list[0][0].dot([1, 1j])},
            comment=f'{result_dict["notes"]}',
            tag='Rabi'
        )


col1, col2, col3 = st.columns(3)
with col1:
    start_len = st.number_input(
        "Start Length (us)", min_value=0.0, value=0.1, step=0.1)

with col2:
    stop_len = st.number_input(
        "Stop Length (us)", min_value=start_len, value=1.0, step=0.1)

with col3:
    steps = st.number_input("Steps:", min_value=1,
                            max_value=1000, value=101, step=1)

st.session_state.config.update(
    [('steps', steps), ('qubit_length_ge', QickSweep1D('lenloop', start_len, stop_len))])

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
    st.session_state.config[config_key] = float(
        new_value) if '.' in new_value else int(new_value)

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
    st.session_state.timerabi = TimeRabi(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.timerabi.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.timerabi.plot(fit=st.session_state.fit_checkbox)

if "timerabi" in st.session_state and st.session_state.timerabi:

    st.session_state.timerabi.plot(fit=st.session_state.fit_checkbox)

    if st.session_state.timerabi_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.timerabi_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            st.session_state.timerabi.save()
            st.success("Data saved successfully!")
    with col2:
        if st.button("SaveLabber"):
            st.session_state.timerabi.save_labber()
            st.success("LabberData saved successfully!")
