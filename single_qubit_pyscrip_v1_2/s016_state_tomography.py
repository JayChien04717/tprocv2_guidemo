# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

# ----- User Library ----- #
from .system_cfg import *
from .system_cfg import DATA_PATH
from .system_tool import get_next_filename_labber, hdf5_generator
from tqdm.auto import tqdm
from .fitting import *
from .yamltool import yml_comment
from IPython.display import display, clear_output

##################
# Define Program #
##################


class StateTomographyCal(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        qubit_ch = cfg["qubit_ch"]

        self.declare_gen(ch=res_ch, nqz=2)
        if self.soccfg["gens"][qubit_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz"], mixer_freq=cfg["qmixer_freq"])
        else:
            self.declare_gen(ch=qubit_ch, nqz=cfg["nqz"])

        self.declare_readout(ch=ro_ch, length=cfg["ro_len"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq_ge"], gen_ch=res_ch
        )

        # readout pulse

        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg["res_freq_ge"],
            phase=cfg["res_phase"],
            gain=cfg["res_gain_ge"],
        )
        # qubit pulses
        self.add_gauss(
            ch=qubit_ch,
            name="ramp",
            sigma=cfg["sigma"],
            length=cfg["sigma"] * 5,
            even_length=True,
        )
        self.add_pulse(
            ch=qubit_ch,
            name="x180",
            style="arb",
            envelope="ramp",
            freq=cfg["f_ge"],
            phase=0,
            gain=cfg["qubit_pi_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="x90",
            style="arb",
            envelope="ramp",
            freq=cfg["f_ge"],
            phase=0,
            gain=cfg["qubit_pi2_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="x90m",
            style="arb",
            envelope="ramp",
            freq=cfg["f_ge"],
            phase=0,
            gain=-cfg["qubit_pi2_gain_ge"],
        )  # R_x(-π/2)
        self.add_pulse(
            ch=qubit_ch,
            name="y90",
            style="arb",
            envelope="ramp",
            freq=cfg["f_ge"],
            phase=+90,
            gain=cfg["qubit_pi2_gain_ge"],
        )
        self.add_pulse(
            ch=qubit_ch,
            name="y90m",
            style="arb",
            envelope="ramp",
            freq=cfg["f_ge"],
            phase=-90,
            gain=cfg["qubit_pi2_gain_ge"],
        )

    def _body(self, cfg):
        qs = cfg["qubit_ch"]
        ros = cfg["ro_ch"]

        prep = cfg.get("prep_state", "0")
        axis = cfg.get("meas_axis", "Z")

        self.send_readoutconfig(ch=ros, name="myro", t=0)

        # ===== (A) 狀態準備 =====
        if prep == "1":
            self.pulse(ch=qs, name="x180", t=0)
            self.delay_auto(0.01)
        elif prep == "+X":
            self.pulse(ch=qs, name="y90", t=0)
            self.delay_auto(0.01)
        elif prep == "-X":
            self.pulse(ch=qs, name="y90m", t=0)
            self.delay_auto(0.01)
        elif prep == "+Y":
            self.pulse(ch=qs, name="x90", t=0)
            self.delay_auto(0.01)
        elif prep == "-Y":
            self.pulse(ch=qs, name="x90m", t=0)
            self.delay_auto(0.01)
        elif prep == "0":
            pass

        # ===== tomography pre-rotation =====
        if axis == "X":
            self.pulse(ch=qs, name="y90m", t=0)  # 量 X
            self.delay_auto(0.01)
        elif axis == "Y":
            self.pulse(ch=qs, name="x90", t=0)  # 量 Y
            self.delay_auto(0.01)
        elif axis == "Z":
            pass

        self.delay_auto(0.05)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[ros], pins=[0], t=cfg["trig_time"])


def calibrate_iq(iq, iq0, iq1):
    """將 complex IQ 投影到 |0>→|1> 軸，回傳 P(|1>)"""
    proj_vec = (iq1 - iq0) / np.abs(iq1 - iq0)
    proj = np.real((iq - iq0) * np.conj(proj_vec))
    scale = np.real((iq1 - iq0) * np.conj(proj_vec))
    return np.clip(proj / scale, 0, 1)


def measure_xyz_moment(soc, soccfg, run_cfg, pyavg, iq0, iq1):
    """固定 prep_state，掃 X/Y/Z，回傳 m=[mx,my,mz] 與原始 IQ 字典"""
    m = {}
    iq_raw = {}
    for axis in ["X", "Y", "Z"]:
        run_cfg["meas_axis"] = axis
        prog = StateTomographyCal(
            soccfg, reps=100, final_delay=run_cfg["relax_delay"], cfg=run_cfg
        )
        iq_list = prog.acquire(soc, rounds=pyavg, progress=False)
        iq = iq_list[0][0].dot([1, 1j])
        p1 = float(calibrate_iq(iq, iq0, iq1))
        m[axis] = 1.0 - 2.0 * p1  # P1 -> <sigma>
        iq_raw[axis] = iq
    return np.array([m["X"], m["Y"], m["Z"]], float), iq_raw


pyavg = 50


run_cfg["prep_state"] = "0"
run_cfg["meas_axis"] = "Z"
prog = StateTomographyCal(
    soccfg, reps=100, final_delay=run_cfg["relax_delay"], cfg=run_cfg
)
iq0 = prog.acquire(soc, rounds=pyavg, progress=False)[0][0].dot([1, 1j])

run_cfg["prep_state"] = "1"
run_cfg["meas_axis"] = "Z"
prog = StateTomographyCal(
    soccfg, reps=100, final_delay=run_cfg["relax_delay"], cfg=run_cfg
)
iq1 = prog.acquire(soc, rounds=pyavg, progress=False)[0][0].dot([1, 1j])


cal_states = [
    ("0", np.array([0.0, 0.0, +1.0])),
    ("1", np.array([0.0, 0.0, -1.0])),
    ("+X", np.array([+1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, +1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
]

M_meas = []
R_true = []
for name, r_true in cal_states:
    run_cfg["prep_state"] = name
    m_vec, _ = measure_xyz_moment(soc, soccfg, run_cfg, pyavg=pyavg, iq0=iq0, iq1=iq1)
    M_meas.append(m_vec)
    R_true.append(r_true)

M_meas = np.vstack(M_meas)
R_true = np.vstack(R_true)


R_ext = np.hstack([R_true, np.ones((R_true.shape[0], 1))])

A = np.zeros((3, 3))
b = np.zeros(3)
for j in range(3):
    x, *_ = np.linalg.lstsq(R_ext, M_meas[:, j], rcond=None)
    A[:, j] = x[:3]
    b[j] = x[3]
A = A.T


pred = (R_true @ A.T) + b  # (N,3)
res = M_meas - pred
print("A=\n", A)
print("b=", b)
print("RMSE per axis =", np.sqrt((res**2).mean(axis=0)))


run_cfg["prep_state"] = "+Y"
m_unknown, iq_xyz = measure_xyz_moment(
    soc, soccfg, run_cfg, pyavg=pyavg, iq0=iq0, iq1=iq1
)


r_est = np.linalg.solve(A, (m_unknown - b))
nr = np.linalg.norm(r_est)
if nr > 1 + 1e-10:
    r_est = r_est / nr * 0.999999


rx, ry, rz = r_est
rho = 0.5 * np.array([[1 + rz, rx - 1j * ry], [rx + 1j * ry, 1 - rz]], dtype=complex)


vals, vecs = np.linalg.eigh(rho)
vals = np.clip(vals, 0, None)
rho_phys = (vecs * (vals / vals.sum() if vals.sum() > 0 else vals)) @ vecs.conj().T

purity = float(np.real(np.trace(rho_phys @ rho_phys)))
print("m_meas =", m_unknown, " | r_est =", r_est, " ||r||=", np.linalg.norm(r_est))
print("rho =\n", rho_phys)
print("eigvals =", np.linalg.eigvalsh(rho_phys), " | purity =", purity)


def plot_density_matrix(
    rho, title=f"state {run_cfg['prep_state']}", phase_cmap="coolwarm", show=True
):
    rho = np.asarray(rho, dtype=complex)
    amp = np.abs(rho)
    phs = np.angle(rho)
    phs_norm = (phs + np.pi) / (2 * np.pi)

    xs, ys = np.meshgrid([0, 1], [1, 0])
    xs = xs.flatten()
    ys = ys.flatten()
    zs = np.zeros_like(xs, dtype=float)
    dx = dy = 0.8
    dz = amp.flatten()

    facecolors = cm.get_cmap(phase_cmap)(phs_norm.flatten())

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xs - 0.4,
        ys - 0.4,
        zs,
        dx,
        dy,
        dz,
        shade=False,
        color=facecolors,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.7,
    )

    ax.set_zlim(0, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$|0\rangle$", r"$|1\rangle$"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$|1\rangle$", r"$|0\rangle$"])
    ax.set_title(title, pad=6)

    # phase colorbar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    mappable = cm.ScalarMappable(norm=norm, cmap=phase_cmap)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("arg", rotation=0, labelpad=8)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$+\pi/2$", r"$\pi$"])

    return fig


plot_density_matrix(rho_phys)
