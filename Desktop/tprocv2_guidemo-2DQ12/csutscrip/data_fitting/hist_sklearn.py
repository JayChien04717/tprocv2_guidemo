from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


def mode_2d_density_max(points, gridsize=100):
    kde = gaussian_kde(points.T)
    x, y = np.linspace(points[:, 0].min(), points[:, 0].max(), gridsize), np.linspace(points[:, 1].min(), points[:, 1].max(), gridsize)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords)
    max_idx = np.argmax(density)
    return grid_coords[:, max_idx] 


def ss_ge(data):
    numbins = 100
    g = np.column_stack([data['g'].real, data['g'].imag])
    e = np.column_stack([data['e'].real, data['e'].imag])

    mu_g = mode_2d_density_max(g)
    mu_e = mode_2d_density_max(e)
    vec = mu_e - mu_g
    theta = np.arctan2(vec[1], vec[0])

    Ig, Qg = g[:, 0], g[:, 1]
    Ie, Qe = e[:, 0], e[:, 1]
    best_theta = theta
    I_tot = np.concatenate((Ie, Ig))
    span = (np.max(I_tot) - np.min(I_tot))/2
    midpoint = (np.max(I_tot) + np.min(I_tot))/2
    xlims = [midpoint - span, midpoint + span]
    ng, _ = np.histogram(Ig, bins=numbins, range=xlims)
    ne, _ = np.histogram(Ie, bins=numbins, range=xlims)
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                      (0.5 * ng.sum() + 0.5 * ne.sum())))
    best_fid = np.max(contrast)

    for theta_i in np.linspace(theta - np.pi/12, theta + np.pi/12, 10):
        Ig_new = Ig * np.cos(theta_i) - Qg * np.sin(theta_i)
        Ie_new = Ie * np.cos(theta_i) - Qe * np.sin(theta_i)
        I_tot_new = np.concatenate((Ie_new, Ig_new))
        span = (np.max(I_tot_new) - np.min(I_tot_new))/2
        midpoint = (np.max(I_tot_new) + np.min(I_tot_new))/2
        xlims = [midpoint - span, midpoint + span]
        ng, _ = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, _ = np.histogram(Ie_new, bins=numbins, range=xlims)
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                          (0.5 * ng.sum() + 0.5 * ne.sum())))
        fid = np.max(contrast)
        if fid > best_fid:
            best_theta = theta_i
            best_fid = fid

    theta = best_theta  # update with optimal rotation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    g_rot = (g) @ R
    e_rot = (e) @ R
    new_data = np.vstack([g_rot, e_rot])

    # GMM fit
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(new_data)
    prob_g = gmm.predict_proba(g_rot)
    prob_e = gmm.predict_proba(e_rot)
    avg_prob_g = prob_g.mean(axis=0)
    avg_prob_e = prob_e.mean(axis=0)

    conf_matrix = np.vstack([avg_prob_g, avg_prob_e])
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    aligned = conf_matrix[:, col_ind]
    fidelity_per_state = np.diag(aligned)
    fidelity_avg = np.mean(fidelity_per_state)

    # GMM center & decision boundary
    mu0, mu1 = gmm.means_
    midpoint = (mu0 + mu1) / 2
    vec = mu1 - mu0
    normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
    pt1 = midpoint + 10 * normal
    pt2 = midpoint - 10 * normal

    # Unrotated scatter
    plt.subplot(2, 2, 1)
    plt.scatter(g[:, 0], g[:, 1], s=2, alpha=0.5, label='g')
    plt.scatter(e[:, 0], e[:, 1], s=2, alpha=0.5, label='e')
    plt.scatter(*mu_g, c='k', marker='o')
    plt.scatter(*mu_e, c='k', marker='o')
    plt.title("Unrotated")
    plt.xlabel("I [ADC levels]")
    plt.ylabel("Q [ADC levels]")
    plt.axis('equal')
    plt.legend()

    # Rotated scatter with boundary
    plt.subplot(2, 2, 2)
    plt.scatter(g_rot[:, 0], g_rot[:, 1], s=2, alpha=0.5, label='g')
    plt.scatter(e_rot[:, 0], e_rot[:, 1], s=2, alpha=0.5, label='e')
    plt.scatter(*mode_2d_density_max(g_rot), color='black', marker='o')
    plt.scatter(*mode_2d_density_max(e_rot), color='black', marker='o')
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r--', label='Decision boundary')
    plt.title(f"Rotated ($\\theta$ = {np.degrees(theta):.2f}°)")
    plt.xlabel("I [ADC levels]")
    plt.ylabel("Q [ADC levels]")
    plt.axis('equal')
    plt.legend()

    # Projected hist with Gauss fit
    plt.subplot(2, 2, 3)
    bins = np.linspace(min(new_data[:, 0]), max(new_data[:, 0]), 100)
    plt.hist(g_rot[:, 0], bins=bins, density=True, alpha=0.6, label='g', color='blue', )
    plt.hist(e_rot[:, 0], bins=bins, density=True, alpha=0.6, label='e', color='orange')
    x = np.linspace(bins[0], bins[-1], 1000)
    def plot_double_gaussian(x_data, color, label):
        gmm1d = GaussianMixture(n_components=2)
        gmm1d.fit(x_data.reshape(-1, 1))
        weights = gmm1d.weights_
        means = gmm1d.means_.flatten()
        covars = gmm1d.covariances_.flatten()
        
        for w, m, v in zip(weights, means, covars):
            plt.plot(x, w * norm.pdf(x, m, np.sqrt(v)), color=color, linestyle='--', alpha=0.8)
        total_pdf = np.sum(w * norm.pdf(x, m, np.sqrt(v)) for w, m, v in zip(weights, means, covars))
        plt.plot(x, total_pdf, color=color, label=label)

    plot_double_gaussian(g_rot[:, 0], 'blue', 'fit g')
    plot_double_gaussian(e_rot[:, 0], 'orange', 'fit e')
    plt.axvline(x=midpoint[0], color='k', linestyle='--')
    plt.text(midpoint[0], plt.ylim()[1]*0.8, f"$\\bar{{F}}_{{ge}}$: {fidelity_avg*100:.1f}%", fontsize=10)
    plt.xlabel("I [ADC levels]")
    plt.ylabel("Probability")
    plt.legend()

    # CDF plot
    plt.subplot(2, 2, 4)
    g_sorted = np.sort(g_rot[:, 0])
    e_sorted = np.sort(e_rot[:, 0])
    g_cdf = np.arange(len(g_sorted)) / len(g_sorted)
    e_cdf = np.arange(len(e_sorted)) / len(e_sorted)
    plt.plot(g_sorted, g_cdf, label='g', color='blue')
    plt.plot(e_sorted, e_cdf, label='e', color='red')
    plt.axvline(x=midpoint[0], color='k', linestyle='--')
    plt.xlabel("I [ADC levels]")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Counts")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    print("Confusion matrix (aligned):\n", np.round(aligned, 3))
    print("Fidelity per state:", np.round(fidelity_per_state, 3))
    print("Average fidelity:", round(fidelity_avg, 3))
    fidelity_error_form = 1 - aligned[0, 1] - aligned[1, 0]
    print("Fidelity (1 - P(e|g) - P(g|e)):", round(fidelity_error_form, 3))
    fidelity_qnd = aligned[0, 0] + aligned[1, 1] - 1
    print(f"QND-style Fidelity (Fgg+Fee-1):   {fidelity_qnd:.3f}")
    return [round(fidelity_avg, 3), midpoint[0], theta]
