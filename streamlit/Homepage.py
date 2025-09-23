import streamlit as st

st.set_page_config(
    page_title="Main page",
    page_icon="👋",
)

st.write("# This is the Single Qubit measurement page 👋")

st.sidebar.success("system configuration")

st.markdown(
    """
    This web interaction page can help you step by step to charaterization qubit parameter

    ### Following below step to fine tune qubit
    - Time Of Flight
    - Resonator spectroscopy
    - Resonator Punch Out
    - Qubit Spectroscopy
    - Qubit Lenght Rabi
    - Qubit Power Rabi
    - T2 Ramsey
    - T1 
    - T2 Echo
    - Single Shot
"""
)
