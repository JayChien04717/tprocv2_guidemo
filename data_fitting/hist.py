import numpy as np
import matplotlib.pyplot as plt
from double_gauss_fit import cal_hist


def histtests(data, plot=True, span=None, verbose=True, title=None, fid_avg=False, fit=False, b_print=False, b_plot=False):
    """
    span: histogram limit is the mean +/- span
    fid_avg: if True, calculate fidelity F by the average mis-categorized e/g; otherwise count
        total number of miscategorized over total counts (gives F^2)
    """
    testtitle = title
    Ig = data[0]
    Qg = data[1]
    Ie = data[2]
    Qe = data[3]
    iqshots = [(Ig, Qg), (Ie, Qe)]
    plot_f = False
    if plot_f:  # 'If' in data.keys():
        plot_f = True
        If = data[4]
        Qf = data[5]

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)} +/- {np.std(np.abs(Ig + 1j*Qg))}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)} +/- {np.std(np.abs(Ig + 1j*Qe))}')
        if plot_f:
            print(
                f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        if b_plot:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
        else:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        if title is not None:
            plt.suptitle(title)
        fig.tight_layout()

        axs[0, 0].scatter(Ig, Qg, label='g', color='b',
                          marker='.', edgecolor='None', alpha=0.2)
        axs[0, 0].scatter(Ie, Qe, label='e', color='r',
                          marker='.', edgecolor='None', alpha=0.2)
        if plot_f:
            axs[0, 0].scatter(If, Qf, label='f', color='g',
                              marker='.', edgecolor='None', alpha=0.2)
        axs[0, 0].plot([xg], [yg], color='k', linestyle=':',
                       marker='o', markerfacecolor='b', markersize=5)
        axs[0, 0].plot([xe], [ye], color='k', linestyle=':',
                       marker='o', markerfacecolor='r', markersize=5)
        if plot_f:
            axs[0, 0].plot([xf], [yf], color='k', linestyle=':',
                           marker='o', markerfacecolor='g', markersize=5)

        # axs[0,0].set_xlabel('I [ADC levels]')
        axs[0, 0].set_ylabel('Q [ADC levels]')
        axs[0, 0].legend(loc='upper right')
        axs[0, 0].set_title('Unrotated', fontsize=14)
        axs[0, 0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg), (xe-xg))
    if plot_f:
        theta = -np.arctan2((yf-yg), (xf-xg))

    """
    Adjust rotation angle
    """
    best_theta = theta
    I_tot = np.concatenate((Ie, Ig))
    span = (np.max(I_tot) - np.min(I_tot))/2
    midpoint = (np.max(I_tot) + np.min(I_tot))/2
    xlims = [midpoint-span, midpoint+span]
    ng, binsg = np.histogram(Ig, bins=numbins, range=xlims)
    ne, binse = np.histogram(Ie, bins=numbins, range=xlims)
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                      (0.5*ng.sum() + 0.5*ne.sum())))
    best_fid = np.max(contrast)
    for theta_i in np.linspace(theta-np.pi/12, theta + np.pi/12, 10):
        Ig_new = Ig*np.cos(theta_i) - Qg*np.sin(theta_i)
        Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta)
        Ie_new = Ie*np.cos(theta_i) - Qe*np.sin(theta_i)
        Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)
        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        I_tot_new = np.concatenate((Ie_new, Ig_new))
        span = (np.max(I_tot_new) - np.min(I_tot_new))/2
        midpoint = (np.max(I_tot_new) + np.min(I_tot_new))/2
        xlims = [midpoint-span, midpoint+span]
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                          (0.5*ng.sum() + 0.5*ne.sum())))
        fid = np.max(contrast)
        if fid > best_fid:
            best_theta = theta_i
            best_fid = fid
    theta = best_theta

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta)

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)} +/- {np.std(np.abs(Ig + 1j*Qg))}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)} +/- {np.std(np.abs(Ig + 1j*Qe))}')
        if plot_f:
            print(
                f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) -
                np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [(xg+xe)/2-span, (xg+xe)/2+span]

    if plot:
        axs[0, 1].scatter(Ig_new, Qg_new, label='g', color='b',
                          marker='.', edgecolor='None', alpha=0.3)
        axs[0, 1].scatter(Ie_new, Qe_new, label='e', color='r',
                          marker='.', edgecolor='None', alpha=0.3)
        if plot_f:
            axs[0, 1].scatter(If_new, Qf_new, label='f', color='g',
                              marker='.', edgecolor='None', alpha=0.3)
        axs[0, 1].plot([xg], [yg], color='k', linestyle=':',
                       marker='o', markerfacecolor='b', markersize=5)
        axs[0, 1].plot([xe], [ye], color='k', linestyle=':',
                       marker='o', markerfacecolor='r', markersize=5)
        if plot_f:
            axs[0, 1].plot([xf], [yf], color='k', linestyle=':',
                           marker='o', markerfacecolor='g', markersize=5)

        axs[0, 1].set_xlabel('I [ADC levels]')
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].set_title('Rotated', fontsize=14)
        axs[0, 1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1, 0].hist(
            Ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = axs[1, 0].hist(
            Ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
        if plot_f:
            nf, binsf, pf = axs[1, 0].hist(
                If_new, bins=numbins, range=xlims, color='g', label='f', alpha=0.5)
        axs[1, 0].set_ylabel('Counts', fontsize=14)
        axs[1, 0].set_xlabel('I [ADC levels]', fontsize=14)
        axs[1, 0].legend(loc='upper right')

    else:
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """fitting the shot gaussian"""

    n_tot_g = [0] * numbins
    n_tot_e = [0] * numbins
    for check_i, data_check in enumerate(iqshots):
        if check_i in [0]:
            n_tot_g += ng
            binsg = binsg
        elif check_i in [1]:
            n_tot_e += ne
            binse = binse

    if fit:
        xmax_g = binsg[np.argmax(n_tot_g)]
        xmax_e = binse[np.argmax(n_tot_e)]

        # a bit stupid but we need to know what the g and e states are to fit the gaussians, and
        # that requires having already looped through all the states once
        popt_lst = []
        for check_i, data_check in enumerate(iqshots):

            I, Q = data_check

            xavg, yavg = np.average(I), np.average(Q)

            I_new = I * np.cos(theta) - Q * np.sin(theta)
            Q_new = I * np.sin(theta) + Q * np.cos(theta)

            n, bins = np.histogram(I_new, bins=numbins, range=xlims)

            idx_g = np.argmin(np.abs(bins[:-1] - xmax_g))
            idx_e = np.argmin(np.abs(bins[:-1] - xmax_e))
            ymax_g = n[idx_g]
            ymax_e = n[idx_e]
            fitparams = [ymax_g, xmax_g, 10, ymax_e, xmax_e, 10]

            popt, pcov = fitter.fit_doublegauss(
                xdata=bins[:-1], ydata=n, fitparams=fitparams)
            popt_lst.append(popt)
            if plot:
                y = fitter.double_gaussian(bins[:-1], *popt)
                y_norm = y

                axs[1, 0].plot(
                    bins[:-1],
                    y_norm,
                    "-",
                    color=default_colors[check_i % len(default_colors)],
                )

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    # this method calculates fidelity as 1-2(Neg + Nge)/N
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                      (0.5*ng.sum() + 0.5*ne.sum())))
    tind = contrast.argmax()
    thresholds.append(binsg[tind])

    if not fid_avg:
        fids.append(contrast[tind])
    else:
        # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
        fids.append(0.5*(1-ng[tind:].sum()/ng.sum() +
                    1-ne[:tind].sum()/ne.sum()))
    if verbose:
        print(f'g correctly categorized: {100*(1-ng[tind:].sum()/ng.sum())}%')
        print(f'e correctly categorized: {100*(1-ne[:tind].sum()/ne.sum())}%')

    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) /
                          (0.5*ng.sum() + 0.5*nf.sum())))
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg:
            fids.append(contrast[tind])
        else:
            fids.append(
                0.5*(1-ng[tind:].sum()/ng.sum() + 1-nf[:tind].sum()/nf.sum()))

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) /
                          (0.5*ne.sum() + 0.5*nf.sum())))
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        if not fid_avg:
            fids.append(contrast[tind])
        else:
            fids.append(
                0.5*(1-ne[tind:].sum()/ne.sum() + 1-nf[:tind].sum()/nf.sum()))

    if plot:
        title = '$\overline{F}_{ge}$' if fid_avg else '$F_{ge}$'
        axs[1, 0].set_title(
            f'Histogram ({title}: {100*fids[0]:.3}%)', fontsize=14)
        axs[1, 0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1, 0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1, 0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1, 1].set_title('Cumulative Counts', fontsize=14)
        axs[1, 1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1, 1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1, 1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            # axs[1, 1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1, 1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1, 1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1, 1].legend()
        axs[1, 1].set_xlabel('I [ADC levels]', fontsize=14)

        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        plt.tight_layout()

    gg = 100*(1-ng[tind:].sum()/ng.sum())
    ge = 100*(ng[tind:].sum()/ng.sum())
    eg = 100*(1-ne[tind:].sum()/ng.sum())
    ee = 100*(ne[tind:].sum()/ng.sum())

    if b_print:
        print(
            f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f}% | {ge:.3f}% |
        ----------------
        | {eg:.3f}% | {ee:.3f}% |
        -----------------
        IQ plane rotated by: {180 / np.pi * theta:.1f}{chr(176)}
        Threshold: {thresholds[0]:.3e}
        Fidelity: {100*fids[0]:.3f}%
        """
        )

    if b_plot:
        axs[0, 2].imshow(np.array([[gg, ge], [eg, ee]]))
        axs[0, 2].set_xticks([0, 1])
        axs[0, 2].set_yticks([0, 1])
        axs[0, 2].set_xticklabels(labels=["|g>", "|e>"])
        axs[0, 2].set_yticklabels(labels=["|g>", "|e>"])
        axs[0, 2].set_ylabel("Prepared", fontsize=14)
        axs[0, 2].set_xlabel("Measured", fontsize=14)
        axs[0, 2].text(0, 0, f"{gg:.1f}%", ha="center", va="center", color="k")
        axs[0, 2].text(1, 0, f"{ge:.1f}%", ha="center", va="center", color="w")
        axs[0, 2].text(0, 1, f"{eg:.1f}%", ha="center", va="center", color="w")
        axs[0, 2].text(1, 1, f"{ee:.1f}%", ha="center", va="center", color="k")
        axs[0, 2].set_title("Fidelities", fontsize=14)

        # lower right text setting
        text_kwargs = dict(ha='center', va='center', fontsize=12)
        axs[1, 2].text(0.45, 0.5, f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f}% | {ge:.3f}% |
        ----------------
        | {eg:.3f}% | {ee:.3f}% |
        -----------------
        IQ plane rotated by: {180 / np.pi * theta:.1f}{chr(176)}
        Threshold: {thresholds[0]:.3e}
        Fidelity: {100*fids[0]:.3f}%
        """, **text_kwargs)
        axs[1, 2].spines['right'].set_color('none')
        axs[1, 2].spines['left'].set_color('none')
        axs[1, 2].spines['bottom'].set_color('none')
        axs[1, 2].spines['top'].set_color('none')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.tight_layout()

    plt.savefig(f'./pic/{testtitle}.png')
    plt.show()
    return fids, thresholds, theta*180/np.pi, popt_lst  # fids: ge, gf, ef
