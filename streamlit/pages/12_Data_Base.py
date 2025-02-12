from single_qubit_pyscrip.system_shotplot import gui_hist
import h5py
import numpy as np
import json
import plotly.graph_objects as go
import streamlit as st
import os
from collections import defaultdict
from typing import Dict, Any

BASE_DIR = st.session_state.datafile


def find_h5_files(base_dir):
    """Scan the directory and construct a nested dictionary of available HDF5 files."""
    file_structure = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for root, _, files in os.walk(base_dir):
        parts = root.split(os.sep)
        if len(parts) < 4:
            continue  # Ensure there are enough levels in the path

        year, month, day = parts[-3], parts[-2], parts[-1]
        for file in files:
            if file.endswith(".h5"):
                file_structure[year][month][day].append(
                    os.path.join(root, file))
    return file_structure


def read_h5_file(file_path: str) -> Dict[str, Any]:
    """Read an HDF5 file and extract relevant data for visualization."""
    data = {}
    with h5py.File(file_path, "r") as f:
        param_grp = f.get("parameter", {})
        data_grp = f.get("data", {})

        x_name, y_name, x_value, y_value = None, None, None, None

        for key in param_grp.keys():
            subgroup = param_grp[key]
            if "x_axis_value" in subgroup:
                x_name = key
                x_value = np.array(subgroup["x_axis_value"])
            elif "y_axis_value" in subgroup:
                y_name = key
                y_value = np.array(subgroup["y_axis_value"])

        if x_name and x_value is not None:
            data["x_name"] = x_name
            data["x_value"] = x_value
        else:
            raise ValueError("No x-axis data found in the HDF5 file.")

        data["y_name"] = y_name if y_name else None
        data["y_value"] = y_value if y_value is not None else None

        z_name = next(iter(data_grp.keys()), None)
        if z_name:
            data["z_name"] = z_name
            data["z_value"] = np.array(data_grp[z_name])
        else:
            raise ValueError("No z-axis data found in the HDF5 file.")

        data["experiment_name"] = f.attrs.get("experiment_name", "")

        data["config"] = json.loads(f.attrs.get("config", "{}"))
        data["result"] = json.loads(f.attrs.get("result", "{}"))

    return data


def plot_data(data: Dict[str, Any], plot_type: str = "magnitude", enable_cursor: bool = True, swap_xy: bool = False):
    """生成 Plotly 圖表，可選擇是否開啟 cursor 以及是否對調 XY 軸"""
    hover_mode = "x unified" if enable_cursor else "closest"  # 是否開啟 cursor

    if data["y_name"] and data["y_value"] is not None:
        X, Y = np.meshgrid(data["x_value"], data["y_value"])
        Z = data["z_value"]

        if swap_xy:
            X, Y = np.meshgrid(data["y_value"], data["x_value"])  # XY 交換
            Z = Z.T  # Z 軸轉置

        if plot_type == "magnitude":
            Z = np.abs(Z)
        elif plot_type == "phase":
            Z = np.angle(Z)
        elif plot_type == "real":
            Z = np.real(Z)
        elif plot_type == "imag":
            Z = np.imag(Z)

        fig = go.Figure(data=go.Heatmap(
            z=Z, x=X[0], y=Y[:, 0], colorscale="Jet",
            hoverongaps=False,
            hovertemplate="x: %{x}<br>y: %{y}<extra></extra>"
        ))

        fig.update_layout(
            title=f"{data['experiment_name']} ({plot_type})",
            xaxis_title=data["y_name"] if swap_xy else data["x_name"],
            yaxis_title=data["x_name"] if swap_xy else data["y_name"],
            plot_bgcolor='rgb(229, 236, 246)',
            hovermode=hover_mode,
            dragmode="pan",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        if hover_mode == "x unified":
            fig.update_xaxes(showspikes=True)
            fig.update_yaxes(showspikes=True)
        else:
            fig.update_xaxes(showspikes=False)
            fig.update_yaxes(showspikes=False)

    else:
        Z = data["z_value"]
        if plot_type == "magnitude":
            Z = np.abs(Z)
        elif plot_type == "phase":
            Z = np.angle(Z)
        elif plot_type == "real":
            Z = np.real(Z)
        elif plot_type == "imag":
            Z = np.imag(Z)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["x_value"], y=Z, mode="lines+markers",
            name=data["z_name"], hoverinfo="x+y",
            line=dict(color="blue", width=2),
            marker=dict(color="blue", size=6)
        ))

        fig.update_layout(
            title=f"{data['experiment_name']}",
            xaxis_title=data["x_name"],
            yaxis_title=data["z_name"],
            plot_bgcolor='rgb(229, 236, 246)',
            hovermode=hover_mode,
            dragmode="pan"
        )
        if hover_mode == "x unified":
            fig.update_xaxes(showspikes=True)
            fig.update_yaxes(showspikes=True)
        else:
            fig.update_xaxes(showspikes=False)
            fig.update_yaxes(showspikes=False)

    return fig


st.set_page_config(page_title="HDF5 Data Viewer", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: rgb(229, 236, 246);
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("HDF5 File Selection")
file_structure = find_h5_files(BASE_DIR) if os.path.exists(BASE_DIR) else {}

years = sorted(file_structure.keys(), reverse=True) if file_structure else []
selected_year = st.sidebar.selectbox("Select Year", years) if years else None

months = sorted(file_structure[selected_year].keys()) if selected_year else []
selected_month = st.sidebar.selectbox(
    "Select Month", months) if months else None

days = sorted(file_structure[selected_year]
              [selected_month].keys()) if selected_month else []
selected_day = st.sidebar.selectbox("Select Day", days) if days else None

files = file_structure[selected_year][selected_month][selected_day] if selected_day else []
file_names = [os.path.basename(f) for f in files]
selected_file_name = st.sidebar.selectbox(
    "Select File", file_names) if file_names else None

selected_file_path = next(
    (f for f in files if os.path.basename(f) == selected_file_name), None)


if selected_file_path:
    data = read_h5_file(selected_file_path)

    st.write("**Config:**", data["config"])
    st.write("**Result:**", data["result"])

    st.divider()
    st.subheader("Data Visualization")

    col1, col2 = st.columns(2)
    with col1:
        enable_cursor = st.checkbox(
            "Enable Cursor Hover", value=True, key="cursor_checkbox")
    with col2:
        swap_xy = st.checkbox("Swap X and Y Axes",
                              value=False, key="swap_checkbox")

    plot_type = st.radio("Select Plot Type",
                         ("magnitude", "phase", "real", "imag"))

    # **繪製圖像**
    fig = plot_data(data, plot_type, enable_cursor, swap_xy)
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "doubleClick": "reset",
        "modeBarButtonsToAdd": ["hoverCompareCartesian"]
    })
