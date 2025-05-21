import streamlit as st
from qick import *
from qick.pyro import make_proxy
from qick import QickConfig
import Pyro4
from single_qubit_pyscrip.system_cfg import *
from single_qubit_pyscrip.system_tool import select_config_idx
# Configure Pyro4
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

# Function to connect to ZCU


def zcu_connect(ip: str):
    soc, soccfg = make_proxy(ns_host=ip, ns_port=8888, proxy_name="myqick")
    return soc, soccfg


# Page title
st.title("Instrument Connection Interface")

# Input field for Ethernet IP
ip_address = st.text_input("Enter Ethernet IP:")

# Button to establish connection
if st.button("Connect"):
    soc, soccfg = zcu_connect(ip_address)
    st.session_state.soc = soc
    st.session_state.soccfg = soccfg
    st.success("Connected successfully!")

# Input field for Data Folder Path
data_path = st.text_input("Input the Datafolder path:")

# Button to set Data Folder
if st.button("Set Datafolder"):
    st.session_state.datafile = data_path
    st.sidebar.success(f"📁 **Data Path:** {st.session_state.datafile}")

# Qubit selection dropdown
st.session_state.QubitIndex = st.sidebar.selectbox(
    "Select Qubit Index:", list(range(0, 7)), index=0
)

st.sidebar.write(f"⚛️ **Selected Qubit:** Q{st.session_state.QubitIndex}")

# Initialize configuration state
if "config_loaded" not in st.session_state:
    st.session_state.config_loaded = False

# Button to load configuration
if st.button("Load Config"):
    st.session_state.hw_cfg = hw_cfg
    st.session_state.readout_cfg = readout_cfg
    st.session_state.qubit_cfg = qubit_cfg
    st.session_state.expt_cfg = expt_cfg
    st.session_state.config_loaded = True
    st.success("Config Loaded Successfully!")

# Sidebar section
st.sidebar.title("Configurations")

# Display Data Path
if "datafile" in st.session_state:
    st.sidebar.write(f"📁 **Data Path:** {st.session_state.datafile}")

# Display SoC configuration
if "soccfg" in st.session_state:
    with st.sidebar.expander("🛠 SoC Configuration"):
        st.write(st.session_state.soccfg)

# Display loaded configurations
if st.session_state.config_loaded:
    with st.sidebar.expander("🖥 Hardware Config"):
        st.json(st.session_state.hw_cfg)

    with st.sidebar.expander("📡 Readout Config"):
        st.json(select_config_idx(st.session_state.readout_cfg,
                idx=int(st.session_state.QubitIndex)))

    with st.sidebar.expander("⚛️ Qubit Config"):
        st.json(select_config_idx(st.session_state.qubit_cfg,
                idx=int(st.session_state.QubitIndex)))

    with st.sidebar.expander("🔬 Experiment Config"):
        st.json(st.session_state.expt_cfg)
