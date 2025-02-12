import h5py
file_path = r'data\2025\02\02-12\000_SingleShot_Q0_3.h5'
with h5py.File(file_path, "r") as f:
    print(f['data'])
