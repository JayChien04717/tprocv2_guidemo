from typing import Any, Dict, Optional
from pprint import pprint
import os
import h5py
import numpy as np
import datetime
import json
import re
from typing import Dict, Any, Union, Optional
import Labber


def update_python_dict(file_path: str, updates: Dict[str, Union[Any, Dict[int, Any]]]) -> None:
    """
    Update dictionary values inside a Python config file while preserving formatting and comments.

    Supports updating specific indices in lists instead of overwriting the entire list.

    Args:
        file_path (str): Path to the Python configuration file.
        updates (Dict[str, Union[Any, Dict[int, Any]]]): Dictionary of updates.
            - Example 1: {"readout_cfg.mixer_freq": 5800}  (Normal update)
            - Example 2: {"qubit_cfg.qubit_freq_ge": {2: 4500}}  (Update list index 2)

    Returns:
        None
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    # Track which dictionary we are modifying
    inside_target: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        modified = False

        # Detect dictionary entry
        for full_key, new_value in updates.items():
            # "qubit_cfg.qubit_freq_ge" → ("qubit_cfg", "qubit_freq_ge")
            dict_name, key = full_key.split(".", 1)

            if stripped.startswith(f"{dict_name} = {{"):
                inside_target = dict_name

            # Only process lines inside the target dictionary
            if inside_target and re.match(rf'^\s*"{key}"\s*:\s*', stripped):
                if isinstance(new_value, dict):  # If updating a list index
                    match = re.search(rf'"{key}"\s*:\s*(\[[^\]]*\])', stripped)
                    if match:
                        # Convert text to Python list
                        old_list = eval(match.group(1))
                        for idx, val in new_value.items():
                            if 0 <= idx < len(old_list):
                                # Update only the specified index
                                old_list[idx] = val
                        new_list_str = str(old_list).replace(
                            "'", "")  # Format correctly
                        line = re.sub(
                            rf'"{key}"\s*:\s*\[[^\]]*\]', f'"{key}": {new_list_str}', line)
                        modified = True
                else:  # Normal value update
                    line = re.sub(
                        rf'("{key}"\s*:\s*)[^,]*', lambda m: f'{m.group(1)}{new_value}', line)

                    modified = True

        new_lines.append(line)

        # Exit dictionary when closing `}`
        if inside_target and stripped == "}":
            inside_target = None

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def select_config_idx(*configs: Dict[str, Any], idx: Optional[int] = None) -> Dict[str, Any]:
    """
    Given multiple configuration dictionaries where values may be lists, select a specific index if applicable.

    Args:
        *configs (Dict[str, Any]): One or more configuration dictionaries.
        idx (Optional[int]): The index to select from list values. If None, keeps the entire list.

    Returns:
        Dict[str, Any]: A merged dictionary with selected values.
    """
    def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
        selected_config = {}
        for key, value in config.items():
            if isinstance(value, list):
                selected_config[key] = value if idx is None else (
                    value[idx] if idx < len(value) else value[0])
            else:
                selected_config[key] = value
        return selected_config

    merged_config = {}
    for config in configs:
        merged_config.update(process_config(config))

    return merged_config


def get_next_filename(base_path: str, exp_name: str, suffix: str = ".h5") -> str:
    """
    Generate a unique filename for an experiment, ensuring no duplicates.

    Args:
        base_path (str): Base directory for saving the file.
        exp_name (str): Experiment name.
        suffix (str): File extension, default is ".h5".

    Returns:
        str: The next available filename.
    """
    today = datetime.date.today()
    year, month, day = today.strftime(
        "%Y"), today.strftime("%m"), today.strftime("%d")
    date_path = f"{month}-{day}"  # Format MM-DD

    experiment_path = os.path.join(base_path, year, month, date_path)
    os.makedirs(experiment_path, exist_ok=True)

    i = 1
    while True:
        fname = f"{exp_name}_{i}{suffix}"
        full_path = os.path.join(experiment_path, fname)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def get_next_filename_labber(dest_path: str, exp_name: str) -> str:
    # make sure dest_path is absolute path
    dest_path = os.path.abspath(dest_path)
    yy, mm, dd = datetime.datetime.today().strftime('%Y-%m-%d').split('-')
    save_path = os.path.join(dest_path, yy, mm, f"Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)

    existing_files = [f for f in os.listdir(save_path) if re.match(
        rf"{re.escape(exp_name)}_\d+\.hdf5", f)]
    next_index = max([int(re.search(r"_(\d+)", f).group(1))
                     for f in existing_files], default=0) + 1

    return os.path.join(save_path, f"{exp_name}_{next_index}")


def saveh5(file_path: str, data_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> None:
    """
    Save experiment data to an HDF5 file.

    Args:
        file_path (str): Path to save the HDF5 file.
        data_dict (Dict[str, Any]): Data to be stored.
        config (Optional[Dict[str, Any]]): Configuration parameters.
        result (Optional[Dict[str, Any]]): Experimental results.
    """
    with h5py.File(file_path, "w") as f:
        param_grp = f.create_group("parameter")
        data_grp = f.create_group("data")

        if "x_name" in data_dict and "x_value" in data_dict:
            x_grp = param_grp.create_group(data_dict["x_name"])
            x_grp.create_dataset("x_axis_value", data=data_dict["x_value"])

        if "y_name" in data_dict and "y_value" in data_dict:
            y_grp = param_grp.create_group(data_dict["y_name"])
            y_grp.create_dataset("y_axis_value", data=data_dict["y_value"])

        if "z_name" in data_dict and "z_value" in data_dict:
            data_grp.create_dataset(
                data_dict["z_name"], data=data_dict["z_value"])
        if "experiment_name" in data_dict:
            f.attrs["experiment_name"] = data_dict["experiment_name"]
        if config:
            f.attrs["config"] = json.dumps(config)
        if result:
            f.attrs["result"] = json.dumps(result)


def saveshot(file_path: str, data_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> None:
    """
    Save data into an HDF5 file.

    Parameters:
    - data (dict): Dictionary containing dataset name and corresponding data.
    - config (dict): Configuration parameters to be stored as attributes.
    - result (dict): Result parameters to be stored as attributes.
    - filename (str): HDF5 file path.
    - data (str): Name of the group where data will be stored.
    """
    with h5py.File(file_path, 'a') as f:
        # Create or get the group
        group = f.require_group('data')

        # Save data into the group
        for key, value in data_dict.items():
            group.create_dataset(
                key, data=value, compression="gzip", overwrite=True)

        if "experiment_name" in data_dict:
            f.attrs["experiment_name"] = data_dict["experiment_name"]
        if config:
            f.attrs["config"] = json.dumps(config)
        if result:
            f.attrs["result"] = json.dumps(result)
# def readh5(file_path: str, option: int = 4) -> Dict[str, Any]:
#     """
#     Read contents from an HDF5 file.

#     Args:
#         file_path (str): Path to the HDF5 file.
#         option (int):
#             1 - Read only "parameter" and "data" groups.
#             2 - Read only config attributes.
#             3 - Read only result attributes.
#             4 - Read all available information (default).

#     Returns:
#         Dict[str, Any]: The requested data based on the specified option.
#     """
#     parameter_dict, data_dict = {}, {}
#     config, result = None, None

#     with h5py.File(file_path, "r") as f:
#         if option in [1, 4]:
#             if "parameter" in f:
#                 param_grp = f["parameter"]
#                 for grp_name in param_grp:
#                     group = param_grp[grp_name]
#                     dataset_names = list(group.keys())
#                     if dataset_names:
#                         parameter_dict[grp_name] = group[dataset_names[0]][:]
#             if "data" in f:
#                 data_grp = f["data"]
#                 for dset_name in data_grp:
#                     data_dict[dset_name] = data_grp[dset_name][:]

#         if option in [2, 4] and "config" in f.attrs:
#             config = json.loads(f.attrs["config"])
#         if option in [3, 4] and "result" in f.attrs:
#             result = json.loads(f.attrs["result"])

#     return {"parameter": parameter_dict, "data": data_dict, "config": config, "result": result}


def read_h5_file(file_path: str) -> Dict[str, Any]:
    """
    Read experiment data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        Dict[str, Any]: Dictionary containing x_name, x_value, y_name 
        (if available), y_value (if available), z_value, config (if available), and result (if available).
    """
    data = {}

    with h5py.File(file_path, "r") as f:
        param_grp = f["parameter"]
        data_grp = f["data"]

        x_name, y_name = None, None
        x_value, y_value = None, None

        # Identify which key contains x_axis_value and y_axis_value
        for key in param_grp.keys():
            subgroup = param_grp[key]
            if "x_axis_value" in subgroup:
                x_name = key
                x_value = subgroup["x_axis_value"][:]
            elif "y_axis_value" in subgroup:
                y_name = key
                y_value = subgroup["y_axis_value"][:]

        # Ensure x_name and x_value exist
        if x_name and x_value is not None:
            data["x_name"] = x_name
            data["x_value"] = np.asarray(x_value)
        else:
            raise ValueError("No x-axis data found in the HDF5 file.")

        # Store y-axis data if available
        if y_name and y_value is not None:
            data["y_name"] = y_name
            data["y_value"] = np.asarray(y_value)
        else:
            data["y_name"] = None
            data["y_value"] = None

        # Extract z-axis data
        z_name = next(iter(data_grp.keys()), None)
        if z_name:
            data["z_name"] = z_name
            data["z_value"] = np.asarray(data_grp[z_name][:])
        else:
            raise ValueError("No z-axis data found in the HDF5 file.")

        # Extract config and result if available
        data["experiment_name"] = json.loads(
            f.attrs["experiment_name"]) if "experiment_name" in f.attrs else None
        data["config"] = json.loads(
            f.attrs["config"]) if "config" in f.attrs else None
        data["result"] = json.loads(
            f.attrs["result"]) if "result" in f.attrs else None

    return data


def hdf5_generator(
        filepath: str,
        x_info: dict, z_info: dict,
        y_info: dict = None, comment=None, tag=None):

    np.bool = bool
    np.float = float
    zdata = z_info['values']
    z_info.update({'complex': True, 'vector': False})
    log_channels = [z_info]
    step_channels = list(filter(None, [x_info, y_info]))

    fObj = Labber.createLogFile_ForData(filepath, log_channels, step_channels)
    if y_info:
        for trace in zdata:
            fObj.addEntry({z_info['name']: trace})
    else:
        fObj.addEntry({z_info['name']: zdata})

    if comment:
        fObj.setComment(comment)
    if tag:
        fObj.setTags(tag)


if __name__ == "__main__":
    # 設定實驗名稱與路徑
    BASE_PATH = "data"
    exp_name = "Experiment_Q1"
    file_path = get_next_filename(BASE_PATH, exp_name)

    data_dict = {
        "x_name": "x_axis",
        "x_value": np.linspace(0, 10, 5),
        "y_name": "y_axis",
        "y_value": np.linspace(0, 20, 4),
        "z_name": "iq_list",
        "z_value": np.outer(np.sin(np.linspace(0, 10, 5)), np.cos(np.linspace(0, 20, 4)))
    }

    config = {
        "ro_ch": 0,
        "res_ch": 0
    }

    result = {
        "T1": "350us",
        "T2": "130us"
    }

    saveh5(file_path, data_dict, config, result)
    print(f"Data saved to: {file_path}")

    # # 讀取 HDF5 檔案
    # loaded_data = readh5(r'F:\CODE\tprocv2_scrip\data\2025\02\02-03\Experiment_Q1_9.h5')
    # print("Loaded Data:", loaded_data)

    # # idx = 3
    # # selected_config = select_config_idx(readout_cfg, qubit_cfg, idx=idx)
    # # pprint(selected_config)

    # QubitIndex = 4
    # config = select_config_idx(readout_cfg, qubit_cfg, idx=QubitIndex)
    # # Update parameters to see TOF pulse with your setup
    # config.update([('res_freq', 7100), ('res_gain', 0.8), ('res_length', 0.5), ('ro_length', 1.5)])
    # pprint(config)

    # **示例：更新 readout_cfg["ro_length"] 和 qubit_cfg["qubit_freq_ge"]**
    # updates = {
    #     'readout_cfg["ro_length"]': "2.0",
    #     'qubit_cfg["qubit_freq_ge"]': "[4500, 4600, 4700, 4800, 4900, 5100]"
    # }

    # update_python_dict(config_file, updates)
    # print("Config updated successfully!")
