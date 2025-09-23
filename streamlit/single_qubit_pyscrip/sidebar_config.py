import streamlit as st

figsize = (8, 4)


def show_sidebar_config(
    hw_cfg, readout_cfg, qubit_cfg, cooling_cfg, expt_cfg, qubit_index=1
):
    """
    Display all config dictionaries in the sidebar using collapsible JSON views.
    """
    st.sidebar.title("Experiment Configuration")
    with st.sidebar.expander("🖥 Hardware Config"):
        st.json(hw_cfg)
    with st.sidebar.expander("📡 Readout Config"):
        st.json(readout_cfg)
    with st.sidebar.expander("⚛️ Qubit Config"):
        st.json(qubit_cfg)
    with st.sidebar.expander("❄️ Cooling Config"):
        st.json(cooling_cfg)
    with st.sidebar.expander("🔬 Experiment Config"):
        st.json(expt_cfg)


def config_update_sidebar(config: dict, cfgs: dict):
    """
    Sidebar UI for selecting and updating a config key-value pair.

    Parameters:
    - config: merged config dictionary (currently used for the experiment)
    - cfgs: dictionary of grouped configs:
        {
            "hw_cfg": ...,
            "readout_cfg": ...,
            "qubit_cfg": ...,
            "cooling_cfg": ...,
            "expt_cfg": ...
        }
    This function updates both the config and its corresponding sub-configs.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Update Config Value")

    # Step 1: Select parameter key
    config_key = st.sidebar.selectbox(
        "Select Config Parameter to Update", list(config.keys())
    )

    # Step 2: Input new value
    new_value_str = st.sidebar.text_input(f"New value for `{config_key}`:")

    # Step 3: Update value when confirmed
    if st.sidebar.button("Update Config"):
        try:
            # Try to parse input value
            if "." in new_value_str:
                new_value = float(new_value_str)
            else:
                new_value = int(new_value_str)
        except ValueError:
            new_value = new_value_str  # fallback to string

        # Update main config
        config[config_key] = new_value

        # Sync all sub-configs if applicable
        for cfg_name, cfg_dict in cfgs.items():
            if config_key in cfg_dict:
                cfg_dict[config_key] = new_value

        st.sidebar.success(f"✅ `{config_key}` updated to `{new_value}`")
        st.rerun()


def sync_param_to_config(param_key: str, value, target_cfg_group: str = "qubit_cfg"):
    """
    精準同步參數到 config 與指定子 config，預設為 qubit_cfg。
    """
    idx = st.session_state.get("QubitIndex", 0)

    # ✅ 更新合併 config
    st.session_state.config[param_key] = value

    # ✅ 更新指定分類 config
    cfg = st.session_state.get(target_cfg_group)
    if cfg is None:
        return  # 該分類不存在

    if isinstance(cfg, list):
        # 確保 index 存在
        while len(cfg) <= idx:
            cfg.append({})
        if not isinstance(cfg[idx], dict):
            cfg[idx] = {}
        cfg[idx][param_key] = value
    elif isinstance(cfg, dict):
        cfg[param_key] = value
