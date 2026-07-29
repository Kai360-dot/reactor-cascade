from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # use on WSL2

import matplotlib.pyplot as plt
import numpy as np

data = Path(__file__).resolve().parent.parent.parent / "data"
# blue: M2d reconstructed joint feasible set
# columns: T1, tau1, T2, tau2, g1, g2, g3, feas
recon = np.loadtxt(data / "recon_joint.csv", delimiter=",", skiprows=1)
X = recon[recon[:, 7] == 1][:, :4]  # keep true-feasible pairs only
# grey: M2a Sobol initialization of the search space (T1, tau1, T2, tau2)
init = np.loadtxt(
    data / "inlet_box.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3)
)
# red: unit-local feasible samples, (T1, tau1) resp. (T2, tau2)
u1 = np.loadtxt(data / "unit1_live.csv", delimiter=",", skiprows=1, usecols=(2, 3))
u2 = np.loadtxt(data / "unit2_live.csv", delimiter=",", skiprows=1, usecols=(2, 3))
unit_pts = {(0, 1): u1, (2, 3): u2}  # only same-unit dim pairs have red points

labels = ["$T_1$ [K]", r"$\tau_1$ [min]", "$T_2$ [K]", r"$\tau_2$ [min]"]
bounds = [(298.15, 353.15), (2, 30), (298.15, 353.15), (2, 30)]
pad = 0.06

NR, NC = 3, 3  # rows and columns per corner block
# two 3x3 corner blocks separated by a spacer column
fig, axs = plt.subplots(
    NR,
    2 * NC + 1,
    figsize=(19, 9),
    gridspec_kw={"width_ratios": [1, 1, 1, 0.25, 1, 1, 1]},
)
for ax in axs[:, NC]:  # spacer column stays empty
    ax.set_visible(False)

blocks = [axs[:, :NC], axs[:, NC + 1 :]]
for b, axb in enumerate(blocks):
    for r in range(NR):
        for c in range(NC):
            ax = axb[r, c]
            if c > r:  # upper triangle stays empty
                ax.set_visible(False)
                continue
            xi, yi = c, r + 1  # panel (r, c): dim c on x vs dim r+1 on y
            (xlo, xhi), (ylo, yhi) = bounds[xi], bounds[yi]
            # painter's order: grey init, red unit-local, blue joint
            ax.plot(init[:, xi], init[:, yi], ".", color="0.75", ms=2)
            if (xi, yi) in unit_pts:  # true iff r,c in (0,0 || 2,2)
                pts = unit_pts[(xi, yi)]
                ax.plot(pts[:, 0], pts[:, 1], ".", color="red", ms=2, alpha=0.4)
            if b == 1:
                ax.plot(X[:, xi], X[:, yi], ".", color="blue", ms=2, alpha=0.4)
            ax.add_patch(
                plt.Rectangle(  # pyright:ignore
                    (xlo, ylo),
                    xhi - xlo,
                    yhi - ylo,
                    fill=False,
                    ls="--",
                    lw=1.2,
                    color="k",
                )
            )
            ax.set_xlim(xlo - pad * (xhi - xlo), xhi + pad * (xhi - xlo))
            ax.set_ylim(ylo - pad * (yhi - ylo), yhi + pad * (yhi - ylo))
            ax.set_xticks(bounds[xi])
            ax.set_yticks(bounds[yi])
            if r == NR - 1:
                ax.set_xlabel(labels[xi])
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(labels[yi])
            else:
                ax.tick_params(labelleft=False)

fig.align_ylabels(axs[:, 0])  # same x-position regardless of tick-label width
fig.align_ylabels(axs[:, NC + 1])
fig.suptitle("CSTR Cascade Linear Simultaneous")
fig.text(0.27, 0.02, "(a) unit-local feasible samples", ha="center")
fig.text(0.76, 0.02, "(b) reconstruction of the joint feasible set", ha="center")

fig.savefig(
    Path(__file__).resolve().parent / "corner_simultaneous.png",
    dpi=300,
    bbox_inches="tight",
)
