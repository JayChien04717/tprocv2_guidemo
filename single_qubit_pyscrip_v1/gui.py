import os
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class H5ViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Viewer")
        self.root.geometry("1200x800")

        # 关闭窗口时，确保程序终止
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 创建主框架
        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # 左侧文件列表框架
        self.left_frame = tk.Frame(
            self.paned_window, width=300, bg="lightgray")
        self.paned_window.add(self.left_frame)

        # 右侧画布框架
        self.right_frame = tk.Frame(self.paned_window, bg="white")
        self.paned_window.add(self.right_frame)

        # 文件列表标签
        self.label = tk.Label(self.left_frame, text="File list",
                              font=("Arial", 14), bg="lightgray")
        self.label.pack(pady=10)

        # 文件列表
        self.file_listbox = tk.Listbox(
            self.left_frame, selectmode=tk.SINGLE, font=("Arial", 12))
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # 选择文件按钮
        self.button_browse = tk.Button(
            self.left_frame, text="Browse File", command=self.load_folder, font=("Arial", 12))
        self.button_browse.pack(pady=10)

        # 右侧画布区域
        self.canvas_frame = tk.Frame(self.right_frame, bg="white")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # 模式选择变量
        self.plot_mode = tk.StringVar(value="Amplitude")

        # 添加模式选择菜单到右侧画布的左上角
        self.mode_menu = tk.OptionMenu(self.canvas_frame, self.plot_mode,
                                       "Amplitude", "Phase", "Real Part", "Imaginary Part",
                                       command=self.update_plot)
        self.mode_menu.pack(side=tk.TOP, anchor="nw",
                            padx=10, pady=10)  # 放置在右侧画布左上角

        # 变量初始化
        self.folder_path = ""
        self.file_map = {}
        self.current_data = None  # 存储当前绘图数据
        self.current_title = ""

    def on_close(self):
        """ 关闭 Tkinter 窗口时，确保 Python 进程结束 """
        self.root.destroy()  # 彻底关闭 GUI

    def load_folder(self):
        self.folder_path = filedialog.askdirectory(title="選擇資料夾")
        if not self.folder_path:
            return

        # 清空檔案列表
        self.file_listbox.delete(0, tk.END)
        self.file_map.clear()

        # 遍歷資料夾中的 .h5 檔案
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".h5"):
                    file_path = os.path.join(root_dir, file)
                    file_name = os.path.basename(file_path)  # 只顯示檔案名稱
                    self.file_map[file_name] = file_path  # 存完整路徑
                    self.file_listbox.insert(tk.END, file_name)

        if not self.file_map:
            messagebox.showinfo("info", "No .h5 File In List")

    def on_file_select(self, event):
        selected_index = self.file_listbox.curselection()
        if not selected_index:
            return

        selected_name = self.file_listbox.get(selected_index[0])  # 取得檔案名稱
        selected_file = self.file_map[selected_name]  # 取得完整路徑
        self.process_h5_file(selected_file)

    def process_h5_file(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'Ig' in f['data'].keys():
                    group = f['data']
                    data_dict = {key: group[key][...] for key in group.keys()}
                    self.plotshot(data_dict=data_dict)
                elif 'Ig' not in f['data'].keys():
                    # 讀取 x 軸和 y 軸的數據
                    x_name = list(f["parameter"].keys())[0]
                    x_value = f[f"parameter/{x_name}/x_axis_value"][:]

                    y_name = list(f["parameter"].keys())[1] if len(
                        f["parameter"].keys()) > 1 else None
                    y_value = f[f"parameter/{y_name}/y_axis_value"][:] if y_name else None

                    # 讀取 z 軸的數據
                    z_name = list(f["data"].keys())[0]
                    z_value = f[f"data/{z_name}"][:]

                    # 存儲數據
                    data_dict = {
                        "x_name": x_name,
                        "x_value": x_value,
                        "y_name": y_name,
                        "y_value": y_value,
                        "z_name": z_name,
                        "z_value": z_value
                    }

                    # 判斷數據維度並繪圖
                    if y_value is None:  # 1D 數據
                        self.plot_1d(data_dict, os.path.basename(file_path))
                    else:  # 2D 數據
                        self.plot_2d(data_dict, os.path.basename(file_path))
        except Exception as e:
            messagebox.showerror("錯誤", f"讀取檔案 {file_path} 時發生錯誤: {str(e)}")

    def update_plot(self, *args):
        """ 根据用户选择的模式，重新绘制图表 """
        if self.current_data:
            self.plot_1d(self.current_data, self.current_title)

    def plot_1d(self, data_dict, title):
        self.clear_canvas()
        fig, ax = plt.subplots(figsize=(8, 6))

        # 获取用户选择的绘图模式
        mode = self.plot_mode.get()

        if mode == "Amplitude":
            y_data = np.abs(data_dict["z_value"])
            ax.set_ylabel("Amplitude")
        elif mode == "Phase":
            y_data = np.angle(data_dict["z_value"])
            ax.set_ylabel("Phase (radians)")
        elif mode == "Real Part":
            y_data = np.real(data_dict["z_value"])
            ax.set_ylabel("Real Part")
        elif mode == "Imaginary Part":
            y_data = np.imag(data_dict["z_value"])
            ax.set_ylabel("Imaginary Part")

        ax.plot(data_dict["x_value"], y_data, label=mode)
        ax.set_xlabel(data_dict["x_name"])
        ax.set_title(title)
        ax.legend()

        self.embed_plot(fig)

    def plot_2d(self, data_dict, title):
        self.clear_canvas()
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(data_dict["x_value"], data_dict["y_value"])
        pcolor = ax.pcolormesh(X, Y, np.abs(
            data_dict["z_value"]), shading='auto')
        fig.colorbar(pcolor, ax=ax, label=data_dict["z_name"])
        ax.set_xlabel(data_dict["x_name"])
        ax.set_ylabel(data_dict["y_name"])
        ax.set_title(title)
        self.embed_plot(fig)

    def plotshot(self, data_dict):
        self.clear_canvas()
        from singleshotplot import gui_hist
        fig = gui_hist(data_dict)
        self.embed_plot(fig)

    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def embed_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# 啟動應用
root = tk.Tk()
app = H5ViewerApp(root)
root.mainloop()
