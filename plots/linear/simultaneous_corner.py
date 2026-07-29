from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # use on WSL2

import matplotlib.pyplot as plt
import numpy as np

path = Path(__file__).resolve().parent.parent.parent / "data" / "global_live.csv"
# columns: crit, feasprob, T1, tau1, T2, tau2
X = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5))

labels = ["$T_1$ [K]", r"$\tau_1$ [min]", "$T_2$ [K]", r"$\tau_2$ [min]"]
bounds = [(298.15, 353.15), (2, 30), (298.15, 353.15), (2, 30)]
pad = 0.06

NR, NC = 3, 3  # number of rows and columns
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for r in range(NR):
    for c in range(NC):
        ax = axs[r, c]
        if c > r:  # upper triangle stays empty
            ax.set_visible(False)
            continue
        xi, yi = c, r + 1  # panel (r, c): dim c on x vs dim r+1 on y
        # print("row:", xi, "col:", yi)
        (xlo, xhi), (ylo, yhi) = bounds[xi], bounds[yi]
        ax.plot(X[:, xi], X[:, yi], ".", color="blue", ms=2, alpha=0.4)
        ax.add_patch(
            plt.Rectangle(  # pyright:ignore
                (xlo, ylo), xhi - xlo, yhi - ylo, fill=False, ls="--", lw=1.2, color="k"
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
fig.suptitle("CSTR Cascade Linear Simultaneous")

fig.savefig(
    Path(__file__).resolve().parent / "corner_simultaneous.png",
    dpi=300,
    bbox_inches="tight",
)
