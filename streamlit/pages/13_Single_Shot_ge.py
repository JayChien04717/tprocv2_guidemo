import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import datetime

from single_qubit_pyscrip.system_tool import (
    select_config_idx,
    saveshot,
    get_next_filename,
    hdf5_generator,
    get_next_filename_labber,
)
from single_qubit_pyscrip.sidebar_config import (
    show_sidebar_config,
    config_update_sidebar,
    sync_param_to_config,
)
from single_qubit_pyscrip.SQ000_SingleShot_prog import (
    SingleShotProgram_g,
    SingleShotProgram_e,
)

st.title("Single Shot ge")
st.session_state.expt_name = "000_SingleShot"
Qubit = "Q" + str(st.session_state.QubitIndex)

# ----- Config Merge -----
st.session_state.config = select_config_idx(
    st.session_state.hw_cfg,
    st.session_state.readout_cfg,
    st.session_state.qubit_cfg,
    st.session_state.cooling_cfg,
    st.session_state.expt_cfg,
    idx=st.session_state.QubitIndex,
)

for key in ["singleshot", "singleshot_fig"]:
    if key not in st.session_state:
        st.session_state[key] = None


class SingleShot_ge:
    def __init__(self, soccfg, cfg):
        self.soccfg = soccfg
        self.cfg = cfg
        self.shot = cfg["shots"]
        self.iq_list_g = None
        self.iq_list_e = None
        self.data = None
        self.data_array = None

    def run(self):
        # ground state
        prog_g = SingleShotProgram_g(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )
        # excited state
        prog_e = SingleShotProgram_e(
            self.soccfg, reps=1, final_delay=self.cfg["relax_delay"], cfg=self.cfg
        )

        st.write("Collecting |g⟩ shots...")
        self.iq_list_g = prog_g.acquire(st.session_state.soc, soft_avgs=1)
        st.write("Collecting |e⟩ shots...")
        self.iq_list_e = prog_e.acquire(st.session_state.soc, soft_avgs=1)

        I_g = self.iq_list_g[0][0].T[0]
        Q_g = self.iq_list_g[0][0].T[1]
        I_e = self.iq_list_e[0][0].T[0]
        Q_e = self.iq_list_e[0][0].T[1]
        self.data_array = np.vstack([I_g + 1j * Q_g, I_e + 1j * Q_e])
        self.data = {"Ig": I_g, "Qg": Q_g, "Ie": I_e, "Qe": Q_e}

        st.session_state.timetag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def plot(self):
        from single_qubit_pyscrip.system_shotplot import gui_hist

        fig = gui_hist(self.data)
        st.session_state.singleshot_fig = fig

    def save(self):
        path = get_next_filename(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
            suffix=".h5",
        )
        data_dict = {"experiment_name": "singleshot"}
        data_dict.update(self.data)
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        saveshot(path, data_dict, result=result_dict)

    def save_labber(self):
        if self.shot is None or self.data_array is None:
            st.error("No data available. Run the experiment first.")
            return
        path = get_next_filename_labber(
            st.session_state.datafile,
            f"{st.session_state.expt_name}_Q{st.session_state.QubitIndex}",
        )
        result_dict = {"notes": str(st.session_state.get("experiment_notes", ""))}
        hdf5_generator(
            filepath=path,
            x_info={"name": "Shot", "unit": "#", "values": np.arange(self.shot)},
            y_info={"name": "State", "unit": "", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": self.data_array},
            comment=result_dict["notes"],
            tag="SingleShot",
        )


# ---- Parameters ----
Shots = st.number_input(
    "Number of shots", min_value=1, max_value=50000, value=5000, step=1
)
st.session_state.config.update({"shots": Shots})
relax_delay = st.number_input(
    "Relaxation time (us)", min_value=1, max_value=1000, value=10, step=1
)
st.session_state.config["relax_delay"] = relax_delay
st.session_state.config["py_avg"] = 1

cool_checkbox = st.checkbox(
    "Apply cool reset", value=st.session_state.get("apply_cool", True)
)
st.session_state.config["apply_cool"] = cool_checkbox
# ---- Sidebar ----
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

# ---- Streamlit Controls ----
if st.button("Run"):
    st.session_state.singleshot = SingleShot_ge(
        st.session_state.soccfg, st.session_state.config
    )
    st.session_state.singleshot.run()
    st.session_state.singleshot.plot()
    st.success("Experiment completed!")

if st.session_state.singleshot and st.session_state.singleshot_fig:
    st.write(f"### Last Measurement Time: {st.session_state.timetag}")
    st.pyplot(st.session_state.singleshot_fig)

st.session_state.experiment_notes = st.text_area(
    "Experiment Notes", placeholder="Note or results..."
)
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        st.session_state.singleshot.save()
        st.success("Data saved successfully!")
with col2:
    if st.button("Save Labber file"):
        st.session_state.singleshot.save_labber()
        st.success("LabberData saved successfully!")
