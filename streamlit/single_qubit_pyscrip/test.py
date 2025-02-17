import os
import datetime
import numpy as np
import Labber
import re
import os
from os.path import isdir, isfile, join
np.float = float
np.bool = bool


def get_next_filename_labber(dest_path: str, exp_name: str) -> str:
    dest_path = os.path.abspath(dest_path)  # 確保 dest_path 為絕對路徑
    yy, mm, dd = datetime.datetime.today().strftime('%Y-%m-%d').split('-')
    save_path = os.path.join(dest_path, yy, mm, f"Data_{mm}{dd}")
    os.makedirs(save_path, exist_ok=True)

    existing_files = [f for f in os.listdir(save_path) if re.match(
        rf"{re.escape(exp_name)}_\d+\.hdf5", f)]
    next_index = max([int(re.search(r"_(\d+)\.hdf5", f).group(1))
                     for f in existing_files], default=0) + 1

    return os.path.join(save_path, f"{exp_name}_{next_index}.hdf5")


def hdf5_generator(
        filepath: str,
        x_info: dict, z_info: dict,
        y_info: dict = None, comment=None, tag=None):

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


def generate_fake_hdf5(filepath):
    freqs = np.linspace(4.0e9, 5.0e9, 101)
    iq_values = np.random.randn(101) + 1j * np.random.randn(101)

    x_info = {'name': 'Frequency', 'unit': "Hz", 'values': freqs}
    z_info = {'name': 'Signal', 'unit': 'a.u.', 'values': iq_values}

    hdf5_generator(filepath, x_info, z_info)
    print(f"Fake HDF5 file created: {filepath}")


def create_labber_datafolder(dest_path: str):
    if not isdir(dest_path):
        raise ValueError('Invalid directory')
    yy, mm, dd = datetime.today().strftime('%Y-%m-%d').split('-')
    save_path = join(dest_path, f'{yy}/{mm}/Data_{mm}{dd}')
    if not isdir(save_path):
        os.makedirs(save_path)
    return save_path


# 測試生成 HDF5 文件
base_path = "./test_data"
exp_name = "FakeExperiment_Q1"
file_path = get_next_filename_labber(base_path, exp_name)

fpts = np.linspace(0, 1, 101)
amps = np.sin(fpts)
hdf5_generator(
    filepath=file_path,
    x_info={'name': 'Frequency', 'unit': "MHz", 'values': fpts/1e3},
    z_info={'name': 'Signal', 'unit': 'a.u.', 'values': amps},
    comment='add 10db attenuator and one 16dB amp', tag=f'OneTone'
)
