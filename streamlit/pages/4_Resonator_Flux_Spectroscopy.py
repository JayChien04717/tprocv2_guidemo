from tqdm import tqdm
import time
import streamlit as st
from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
from single_qubit_pyscrip.system_tool import hdf5_generator, get_next_filename_labber
from single_qubit_pyscrip.SQ002_res_spec_ge import SingleToneSpectroscopyProgram
from qick.asm_v2 import QickSpan, QickSweep1D
import matplotlib.pyplot as plt
import numpy as np
import datetime
import Labber
from tqdm import tqdm

st.title("Resonator OneTone Spectroscopy")

yoko_connect = st.text_input("Enter yoko address:")
if st.button('Connect to yoko'):
    client = Labber.connectToServer('localhost', timeout=86400)
    st.session_state.yoko_global = client.connectToInstrument(
        'Yokogawa GS200 DC Source', dict(
            interface='USB', address=str(yoko_connect))
    )
    yoko_global_cfg = {
        'Current - Sweep rate': 10e-6,
        'Function': 'Current',
        'Range (I)': '10 mA',
        'Output': True,
    }
    st.session_state.yoko_global.startInstrument()
    st.session_state.yoko_global.setInstrConfig(dValues=yoko_global_cfg)
    st.success("Connected successfully!")

# ----- Experiment Configurations ----- #
st.session_state.expt_name = "002_onetone_flux"
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
for key in ["onetoneflux", "config", "onetoneflux_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class ResonatorOnetoneFlux:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.freqs = None
        self.current = None

    def run(self, reps):
        prog = SingleToneSpectroscopyProgram(
            self.soccfg, reps=reps, final_delay=self.cfg['relax_delay'], cfg=self.cfg)
        py_avg = self.cfg['py_avg']

        # Calculate total steps and ensure it's valid
        total_steps = len(np.linspace(st.session_state.start_current,
                                      st.session_state.stop_current, st.session_state.current_steps))

        # Ensure total_steps is not zero
        if total_steps == 0:
            st.error(
                "Total steps is zero. Please check the current range or step size.")
            return

        self.current = np.linspace(st.session_state.start_current,
                                   st.session_state.stop_current, st.session_state.current_steps)
        self.iq_list = []

        # Initialize Streamlit progress bar
        progress_bar = st.progress(0)

        # Initialize tqdm progress bar and manually update Streamlit progress bar
        for idx, i in tqdm(enumerate(self.current), total=total_steps, desc="Running", unit="step", dynamic_ncols=True):
            st.session_state.yoko_global.setValue(
                sQuant='Current', value=i, rate=10e-6)

            iq = prog.acquire(
                st.session_state.soc, soft_avgs=py_avg, progress=False)

            self.iq_list.append(iq[0][0].dot([1, 1j]))
            # Calculate the current progress
            progress = (idx + 1) / total_steps
            progress_bar.progress(progress)  # Update Streamlit's progress bar

        self.freqs = prog.get_pulse_param("res_pulse", "freq", as_array=True)

    def plot(self):
        plt.rcParams.update({'font.size': 14})
        if self.iq_list is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.pcolormesh(self.current*1e-3, self.freqs,
                          np.abs(self.iq_list).T, label="Magnitude",)
            ax.set_xlabel("Current (mA)")
            ax.set_ylabel("Frequency (MHz)")
            ax.set_title("Resonator OneTone Flux Spectroscopy")
            ax.legend()

            # Save to session state
            st.session_state.onetoneflux_fig = fig
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
            "experiment_name": "res_flux_spec_ge",
            "x_name": "Current (mA)",
            "x_value": self.current,
            "y_name": "Frequency (MHz)",
            "y_value": self.freqs,
            "z_name": "ADC unit (a.u)",
            "z_value": np.array(self.iq_list).T
        }
        result_dict = {"notes": str(
            st.session_state.get("experiment_notes", ""))}
        saveh5(file_path, data_dict, result=result_dict)

    def save_labber(self):
        """
        Save experimental data into an HDF5 file.
        """
        if self.freqs is None or self.iq_list is None:
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
            x_info={'name': 'Current', 'unit': "A",
                    'values': self.current},
            y_info={'name': 'Frequency', 'unit': "Hz",
                    'values': self.freqs*1e9},
            z_info={'name': 'Signal', 'unit': 'a.u.',
                    'values':  np.array(self.iq_list).T},
            comment=f'{result_dict["notes"]}',
            tag='OneTone'
        )


col1, col2, col3 = st.columns(3)


def number_input_mA(label, min_mA, max_mA, default_mA, step_mA):
    """Force the input to use mA as the unit while internally converting it to A."""
    input_mA = st.number_input(
        f"{label} (mA)",
        min_value=min_mA,
        max_value=max_mA,
        value=default_mA,
        step=step_mA,
        format="%.3f"  # Display values with three decimal places
    )
    return input_mA / 1000  # Convert mA to A for internal use


# Create three columns for user input
with col1:
    st.session_state.start_current = number_input_mA(
        "Start Current", min_mA=-10.0, max_mA=10.0, default_mA=0.0, step_mA=0.001)

with col2:
    st.session_state.stop_current = number_input_mA(
        "Stop Current", min_mA=-10.0, max_mA=10.0, default_mA=0.0, step_mA=0.001)

with col3:
    st.session_state.current_steps = st.number_input("Current Steps:", min_value=1,
                                                     max_value=10000, value=101, step=1)


col1, col2, col3 = st.columns(3)
with col1:
    start_freq = st.number_input(
        "Start Frequency (MHz)", min_value=0, value=4000, step=1)
with col2:
    stop_freq = st.number_input(
        "Stop Frequency (MHz)", min_value=start_freq, value=9000, step=1)
with col3:
    steps = st.number_input("Steps:", min_value=1,
                            max_value=10000, value=101, step=1)

st.session_state.config.update(
    [('steps', steps), ('res_freq_ge', QickSweep1D('freqloop', start_freq, stop_freq))])

# **Soft Average Configuration**
pyavg = st.number_input("Soft average #:", min_value=1,
                        max_value=10000, value=100, step=1)
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


if st.button("Run"):
    st.session_state.onetoneflux = ResonatorOnetoneFlux(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.onetoneflux.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.onetoneflux.plot()


if "onetoneflux" in st.session_state and st.session_state.onetoneflux:
    st.session_state.onetoneflux.plot()
    if st.session_state.onetoneflux_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.onetoneflux_fig)

    st.session_state.experiment_notes = st.text_area(
        "Experiment Notes", placeholder="Note or results...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            st.session_state.onetoneflux.save()
            st.success("Data saved successfully!")
    with col2:
        if st.button("SaveLabber"):
            st.session_state.onetoneflux.save_labber()
            st.success("LabberData saved successfully!")
