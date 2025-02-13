# from system_tool import readh5
from pprint import pprint
import h5py

loaded_data = (
    r'C:\Users\SQC\Desktop\tprocv2_scrip-main\streamlit\data\2025\02\02-13\000_SingleShot_Q0_2.h5')
with h5py.File(loaded_data, "r") as f:

    print(f.keys())


def getparameter(file_path):
    """
    讀取 HDF5 檔案中 "parameter" 群組的所有子群組名稱（key）。

    Parameters:
      file_path (str): HDF5 檔案路徑

    Returns:
      list: "parameter" 群組中所有子群組的名稱。
    """
    with h5py.File(file_path, "r") as f:
        if "parameter" not in f:
            raise KeyError(f"'parameter' group 不存在於 {file_path}")
        param_grp = f["parameter"]
        # 取得 "parameter" 群組下所有子群組名稱
        keys = list(param_grp.keys())
    return keys


def getparametervalue(file_path):
    """
    讀取 HDF5 檔案中 "parameter" 群組內每個子群組的第一個 dataset 數值。

    Parameters:
      file_path (str): HDF5 檔案路徑

    Returns:
      dict: 一個字典，其 key 為子群組名稱，value 為該子群組內第一個 dataset 的數據。
    """
    values = {}
    with h5py.File(file_path, "r") as f:
        if "parameter" not in f:
            raise KeyError(f"'parameter' group 不存在於 {file_path}")
        param_grp = f["parameter"]
        for key in param_grp:
            group = param_grp[key]
            dataset_names = list(group.keys())
            if dataset_names:
                # 讀取第一個 dataset 的數值
                ds_name = dataset_names[0]
                values[key] = group[ds_name][:]
            else:
                values[key] = None  # 沒有 dataset 時，設定為 None
    return values


# # 取得 parameter 的 key 列表
# keys = getparameter(
#     r'C:\Users\SQC\Desktop\QICK\jay scrip\tprocv2_demos-main\data\2025\02\02-04\002b_res_pumchout_ge_Q2_1.h5')
# print("Parameter keys:", keys)

# 取得 parameter 的數值
# param_values = getparametervalue(
#     r'C:\Users\SQC\Desktop\QICK\jay scrip\tprocv2_demos-main\data\2025\02\02-04\002b_res_pumchout_ge_Q2_1.h5')
# print("Parameter values:", param_values)
