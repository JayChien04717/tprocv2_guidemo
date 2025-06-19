import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import datetime

from single_qubit_pyscrip.system_tool import select_config_idx, saveh5, get_next_filename
import single_qubit_pyscrip.fitting as fitter
from single_qubit_pyscrip.SQ004_time_rabi import LengthRabiProgram


st.title("Qubit Length Rabi Experiment")

# ---- Experiment Configurations ---- #
st.session_state.expt_name = "004_length_rabi"
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
for key in ["rabi", "config", "rabi_fig", "fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class LengthRabi:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.iq_list = None
        self.t = None

    def run(self, reps):
        final_delay = self.cfg.get("relax_delay", 10.0)  # Default if not set
        prog = LengthRabiProgram(
            self.soccfg, reps=reps, final_delay=final_delay, cfg=self.cfg)

        py_avg = self.cfg['py_avg']
        self.iq_list = prog.acquire(st.session_state.soc, soft_avgs=py_avg)
        self.t = prog.get_pulse_param("qubit_pulse", "length", as_array=True)

    def plot(self, fit=False):
        if self.iq_list is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(self.t, np.abs(
                self.iq_list[0][0].dot([1, 1j])), label="Magnitude")
            ax.set_xlabel("Pulse Length (ns)")
            ax.set_ylabel("ADC unit (a.u)")
            ax.set_title("Qubit Length Rabi Experiment")
            ax.legend()

            if fit:
                pOpt, pCov = fitter.fit_rabi(
                    self.t, np.abs(self.iq_list[0][0].dot([1, 1j])))
                rabi_period = pOpt[1]
                ax.plot(self.t, fitter.rabi_func(self.t, *pOpt),
                        label=f"Fit Period = {rabi_period:.1f} ns")
                ax.legend()

            # Save to session state
            st.session_state.rabi_fig = fig
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
            "experiment_name": "length_rabi",
            "x_name": "Pulse Length (ns)",
            "x_value": self.t,
            "z_name": "ADC unit (a.u)",
            "z_value": self.iq_list[0][0].dot([1, 1j])
        }

        saveh5(file_path, data_dict)


# ----- UI for Experiment Parameters ----- #
start_length = st.number_input(
    "Start Length (ns)", min_value=0, value=0, step=1)
stop_length = st.number_input(
    "Stop Length (ns)", min_value=start_length, value=200, step=1)
steps = st.number_input("Steps:", min_value=1,
                        max_value=1000, value=101, step=1)

st.session_state.config.update(
    [('steps', steps), ('qubit_length_ge',
                        np.linspace(start_length, stop_length, steps))]
)

pyavg = st.number_input("Soft average #:", min_value=1,
                        max_value=10000, value=10, step=1)
st.session_state.config['py_avg'] = pyavg


######################################
# ---- Sidebar Configurations ---- #
######################################
st.sidebar.title("Experiment Configuration")

qubit_index = int(st.session_state.get("QubitIndex", 1))

with st.sidebar.expander("🖥 Hardware Config"):
    st.json(st.session_state.hw_cfg)

with st.sidebar.expander("📡 Readout Config"):
    st.json(select_config_idx(st.session_state.readout_cfg, idx=qubit_index))

with st.sidebar.expander("⚛️ Qubit Config"):
    st.json(select_config_idx(st.session_state.qubit_cfg, idx=qubit_index))

with st.sidebar.expander("🔬 Experiment Config"):
    st.json(st.session_state.expt_cfg)

# Dropdown to update configuration
config_key = st.sidebar.selectbox(
    "Select Config Parameter to Update", list(st.session_state.config.keys()))
new_value = st.sidebar.text_input(f"New value for {config_key}:")

if st.sidebar.button("Update Config"):
    st.session_state.config[config_key] = eval(
        new_value) if new_value.isnumeric() else new_value
    st.session_state.hw_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.hw_cfg}
    st.session_state.readout_cfg = {k: v for k, v in st.session_state.config.items(
    ) if k in st.session_state.readout_cfg}
    st.session_state.qubit_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.qubit_cfg}
    st.session_state.expt_cfg = {
        k: v for k, v in st.session_state.config.items() if k in st.session_state.expt_cfg}
    st.rerun()


###############################
# ---- Streamlit Actions ---- #
###############################

fit_checkbox = st.checkbox(
    "Fit Data", value=st.session_state.get("fit_checkbox", False))
st.session_state.fit_checkbox = fit_checkbox

if st.button("Run"):
    st.session_state.rabi = LengthRabi(
        st.session_state.soccfg, st.session_state.config)
    st.session_state.rabi.run(reps=st.session_state.config['reps'])
    st.success("Experiment completed!")
    st.session_state.rabi.plot(fit=st.session_state.fit_checkbox)

if "rabi" in st.session_state and st.session_state.rabi:
    st.session_state.rabi.plot(fit=st.session_state.fit_checkbox)

    if st.session_state.rabi_fig:
        st.write(f"### Last Measurement Time: {st.session_state.timetag}")
        st.pyplot(st.session_state.rabi_fig)

    if st.button("Save"):
        st.session_state.rabi.save()
        st.success("Data saved successfully!")
