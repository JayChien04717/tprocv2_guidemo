
import scipy as sp
import numpy as np


# fit data to a 2D Gaussian
def Gaussian2D(xy, xo, yo, sigma):
    x, y = xy
    amplitude = 1 / (2 * np.pi * sigma**2)
    g = amplitude * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma**2))

    return g.ravel()


def DualGaussian2D(xy, xo0, yo0, sigma0, xo1, yo1, sigma1, a_ratio):
    A, B = a_ratio, 1 - a_ratio
    return A * Gaussian2D(xy, xo0, yo0, sigma0) + B * Gaussian2D(xy, xo1, yo1, sigma1)


X_LEN, Y_LEN = 100, 100


def fit_data(X: np.ndarray, Y: np.ndarray, D: np.ndarray):
    # initial guess
    max_id = np.argmax(D)
    x_max, y_max = X[max_id], Y[max_id]
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    sigma = (np.std(X) + np.std(Y)) / 2

    x0 = x_max
    y0 = y_max
    x1 = 2 * x_mean - x_max
    y1 = 2 * y_mean - y_max

    p0 = (x0, y0, 0.5 * sigma, x1, y1, 0.5 * sigma, 0.8)

    # fit
    param, _ = sp.optimize.curve_fit(
        DualGaussian2D,
        (X, Y),
        D,
        p0=p0,
        bounds=(
            [
                x0 - 2.5 * sigma,
                y0 - 2.5 * sigma,
                0,
                x1 - 5 * sigma,
                x1 - 5 * sigma,
                0,
                0.6,
            ],
            [
                x0 + 2.5 * sigma,
                y0 + 2.5 * sigma,
                sigma,
                x1 + 5 * sigma,
                y1 + 5 * sigma,
                sigma,
                1.0,
            ],
        ),
    )

    return param


# change coordinate to make two fit center align with x-axis
def rotate(x, y, angle):
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)

    return x_rot, y_rot


def rotate_data(X0, Y0, param0, X1, Y1, param1):
    center00 = np.array([param0[0], param0[1]])
    center01 = np.array([param0[3], param0[4]])
    center10 = np.array([param1[0], param1[1]])
    center11 = np.array([param1[3], param1[4]])

    angle = np.arctan2(center10[1] - center00[1], center10[0] - center00[0])

    X0_rot, Y0_rot = rotate(X0, Y0, angle)
    X1_rot, Y1_rot = rotate(X1, Y1, angle)

    param0_rot = list(param0)
    param0_rot[0], param0_rot[1] = rotate(center00[0], center00[1], angle)
    param0_rot[3], param0_rot[4] = rotate(center01[0], center01[1], angle)

    param1_rot = list(param1)
    param1_rot[0], param1_rot[1] = rotate(center10[0], center10[1], angle)
    param1_rot[3], param1_rot[4] = rotate(center11[0], center11[1], angle)

    return angle, X0_rot, Y0_rot, param0_rot, X1_rot, Y1_rot, param1_rot


# calculate density of points by kde
def calculate_density(X, Y, bandwidth: float = 0.1):
    return sp.stats.gaussian_kde([X, Y], bw_method=bandwidth)([X, Y])


def calculate_histogram(X0, X1, BIN_NUM):
    bins = np.linspace(min(min(X0), min(X1)), max(max(X0), max(X1)), BIN_NUM)
    dbin = bins[1] - bins[0]
    hist0, _ = np.histogram(X0, bins=bins, density=True)
    hist1, _ = np.histogram(X1, bins=bins, density=True)

    return bins, hist0, hist1


def calculate_threshold(bins, hist0, hist1):
    contrast = np.abs(
        (np.cumsum(hist0) - np.cumsum(hist1)) /
        (0.5 * hist0.sum() + 0.5 * hist1.sum())
    )

    tind = np.argmax(contrast)
    threshold = (bins[tind] + bins[tind + 1]) / 2
    fid = contrast[tind]

    return fid, threshold


def Gaussian1D(x, xo, sigma):
    amplitude = 1 / (sigma * np.sqrt(2 * np.pi))
    g = amplitude * np.exp(-((x - xo) ** 2) / (2 * sigma**2))

    return g


def DualGaussian1D(x, xo0, sigma0, xo1, sigma1, a_ratio):
    A, B = a_ratio, 1 - a_ratio
    return A * Gaussian1D(x, xo0, sigma0) + B * Gaussian1D(x, xo1, sigma1)


def Gaussian1D(x, xo, sigma, amplitude):
    """1D Gaussian with specified amplitude."""
    g = amplitude * np.exp(-((x - xo) ** 2) / (2 * sigma**2))
    return g


def DualGaussian1D(x, xo0, sigma0, amp0, xo1, sigma1, amp1):
    """Sum of two 1D Gaussians with specified amplitudes."""
    g0 = Gaussian1D(x, xo0, sigma0, amp0)
    g1 = Gaussian1D(x, xo1, sigma1, amp1)
    return g0 + g1


def cal_hist(data, BIN_NUM=50):
    Ig, Qg, Ie, Qe = data[0], data[1], data[2], data[3]
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)

    theta = -np.arctan2((ye - yg), (xe - xg))

    """Rotate the IQ data"""
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    bins, hist0, hist1 = calculate_histogram(Ig_new, Ie_new, BIN_NUM)
    fidelity, threshold = calculate_threshold(bins, hist0, hist1)
    x = np.linspace(min(bins), max(bins), BIN_NUM)

    fit_g = DualGaussian1D(
        x,
        param_g_new[0], param_g_new[2], param_g_new[1],
        param_g_new[3], param_g_new[5], param_g_new[4],
    ) * (x[1] - x[0])
    fit_e = DualGaussian1D(
        x,
        param_e_new[0], param_e_new[2], param_e_new[1],
        param_e_new[3], param_e_new[5], param_e_new[4],
    ) * (x[1] - x[0])
    return fit_g, fit_e


def plot_histogram2(bins, hist0, hist1, param0, param1, threshold):
    _, ax = plt.subplots()

    # plot histogram and threshold
    ax.bar(bins[:-1], hist0, width=bins[1] -
           bins[0], alpha=0.5, label="State 0")
    ax.bar(bins[:-1], hist1, width=bins[1] -
           bins[0], alpha=0.5, label="State 1")
    ax.axvline(threshold, color="black", linestyle="--", label="Threshold")

    # plot fitting curve
    def Gaussian1D(x, xo, sigma):
        amplitude = 1 / (sigma * np.sqrt(2 * np.pi))
        g = amplitude * np.exp(-((x - xo) ** 2) / (2 * sigma**2))

        return g

    def DualGaussian1D(x, xo0, sigma0, xo1, sigma1, a_ratio):
        A, B = a_ratio, 1 - a_ratio
        return A * Gaussian1D(x, xo0, sigma0) + B * Gaussian1D(x, xo1, sigma1)

    x = np.linspace(min(bins), max(bins), BIN_NUM)

    fit0 = DualGaussian1D(x, param0[0], param0[2], param0[3], param0[5], param0[6]) * (
        x[1] - x[0]
    )
    fit1 = DualGaussian1D(x, param1[0], param1[2], param1[3], param1[5], param1[6]) * (
        x[1] - x[0]
    )

    ax.plot(x, fit0, color="blue", label="Fit 0")
    ax.plot(x, fit1, color="red", label="Fit 1")

    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    # os.chdir(r'D:\Jay PhD\Code\data_fitting\data')
    sys.path.append(r'C:\Program Files\Keysight\Labber\Script')
    import Labber
    import matplotlib.pyplot as plt
    np.bool = bool
    np.float = np.float64

    shot = r"D:\Jay PhD\5q data\shot\q1 single shot 30000 gain 1.5us.hdf5"
    shot = Labber.LogFile(shot)
    g = shot.getData()[0]
    e = shot.getData()[1]
    Ig, Qg = g.real, g.imag
    Ie, Qe = e.real, e.imag

    BIN_NUM = 200
    D0 = calculate_density(Ig, Qg)
    D1 = calculate_density(Ie, Qe)

    param0 = fit_data(Ig, Qg, D0)
    param1 = fit_data(Ie, Qe, D1)
    print(f"Fit 0: {param0}")
    print(f"Fit 1: {param1}")

    angle, rX0, rY0, rparam0, rX1, rY1, rparam1 = rotate_data(
        Ig, Qg, param0, Ie, Qe, param1
    )
    bins, hist0, hist1 = calculate_histogram(rX0, rX1, 200)
    fidelity, threshold = calculate_threshold(bins, hist0, hist1)
    plot_histogram2(bins, hist0, hist1, rparam0, rparam1, threshold)
    print(f"Angle: {angle: .2f} (rad)")
    print(f"Threshold: {threshold: .2f}")
    print(f"Fidelity: {fidelity: .3f}")
