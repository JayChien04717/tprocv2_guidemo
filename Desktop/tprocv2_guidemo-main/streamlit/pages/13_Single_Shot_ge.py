import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveshot, get_next_filename
from single_qubit_pyscrip.system_tool import hdf5_generator, get_next_filename_labber
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ000_SingleShot_prog import SingleShotProgram_g, SingleShotProgram_e
from qick.asm_v2 import QickSpan, QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime
import Labber
from tqdm import tqdm

st.title("Single Shot ge")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "000_SingleShot"
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
for key in ["singleshot", "config", "singleshot_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class SingleShot_ge:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.shot = cfg['shots']

    def run(self, shot_f=False):
        shot_g = SingleShotProgram_g(
            self.soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        shot_e = SingleShotProgram_e(
            self.soccfg, reps=1, final_delay=self.cfg['relax_delay'], cfg=self.cfg)

        st_progress = st.progress(0)
        st.write('shot g')
        self.iq_list_g = shot_g.acquire(
            st.session_state.soc, soft_avgs=1, progress=True, st_progress=st_progress)
        st_progress = st.progress(0)
        st.write('shot g')
        self.iq_list_e = shot_e.acquire(
            st.session_state.soc, soft_avgs=1, progress=True, st_progress=st_progress)

        I_g = self.iq_list_g[0][0].T[0]
        Q_g = self.iq_list_g[0][0].T[1]
        I_e = self.iq_list_e[0][0].T[0]
        Q_e = self.iq_list_e[0][0].T[1]
        self.data_array = self.data_array = np.vstack(
            [(I_g + 1j * Q_g), (I_e + 1j * Q_e)])

        self.data = {'Ig': I_g, 'Qg': Q_g,
                     'Ie': I_e, 'Qe': Q_e}

    def plot(self, fit=False):
        from single_qubit_pyscrip.system_shotplot import gui_hist
        plt.rcParams.update({'font.size': 14})
        if self.shot is not None:
            fig, ax = plt.subplots(figsize=(14, 7))
            fig = gui_hist(self.data)
            st.session_state.singleshot_fig = fig
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
            "experiment_name": "singleshot",
        }
        data_dict.update(self.data)
        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}

        saveshot(file_path, data_dict, result=result_dict)

    def save_labber(self):
        """
        Save experimental data into an HDF5 file.
        """
        if self.shot is None:
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
            x_info={'name': 'Shot', 'unit': "#",
                    'values':  np.arange(self.shot)},
            y_info={'name': 'State', 'unit': "",
                    'values': [0, 1]},
            z_info={'name': 'Signal', 'unit': 'a.u.',
                    'values': self.data_array},
            comment=f'{result_dict["notes"]}',
            tag='SingleShot'
        )


Shots = st.number_input(" \# of shots:", min_value=1,
                        max_value=50000, value=5000, step=1)

st.session_state.config.update([('shots', Shots)])

# **Soft Average Configuration**


relax_delay = st.number_input("relaxatoin time (us):", min_value=1,
                              max_value=1000, value=10, step=1)

st.session_state.config['relax_delay'] = relax_delay
st.session_state.config['py_avg'] = 1

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
    st.session_state.singleshot = SingleShot_ge(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.singleshot.run()
    st.success("Experiment completed!")
    st.session_state.singleshot.plot()

if "singleshot" in st.session_state and st.session_state.singleshot:

    st.session_state.singleshot.plot()

    if st.session_state.singleshot_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.singleshot_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            st.session_state.singleshot.save()
            st.success("Data saved successfully!")
    with col2:
        if st.button("Save Labber file"):
            st.session_state.singleshot.save_labber()
            st.success("Data saved successfully!")
