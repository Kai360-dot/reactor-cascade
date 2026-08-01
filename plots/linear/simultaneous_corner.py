import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # use on WSL2

import numpy as np

PLOTS = Path(__file__).resolve().parent.parent  # .../reactor_cascade/plots
sys.path.insert(0, str(PLOTS))  # make `tools` importable from any cwd

from tools.corner import corner_plot

DATA = PLOTS.parent / "data"
# blue: M2d reconstructed joint feasible set
# columns: T1, tau1, T2, tau2, g1, g2, g3, feas
recon = np.loadtxt(DATA / "recon_joint.csv", delimiter=",", skiprows=1)
X = recon[recon[:, 7] == 1][:, :4]  # keep true-feasible pairs only
# grey: M2a Sobol initialization of the search space (T1, tau1, T2, tau2)
init = np.loadtxt(
    DATA / "inlet_box.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3)
)
# red: unit-local feasible samples, (T1, tau1) resp. (T2, tau2)
u1 = np.loadtxt(DATA / "unit1_live.csv", delimiter=",", skiprows=1, usecols=(2, 3))
u2 = np.loadtxt(DATA / "unit2_live.csv", delimiter=",", skiprows=1, usecols=(2, 3))


def main():
    corner_plot(
        [  # painter's order: grey init, red unit-local, blue joint
            dict(data=init, color="0.75"),
            dict(data={(0, 1): u1, (2, 3): u2}, color="red", alpha=0.4),
            dict(data=X, color="blue", alpha=0.4, blocks=[1]),
        ],
        labels=["$T_1$ [K]", r"$\tau_1$ [min]", "$T_2$ [K]", r"$\tau_2$ [min]"],
        bounds=[(298.15, 353.15), (2, 30), (298.15, 353.15), (2, 30)],
        nblocks=2,
        captions=[
            "(a) unit-local feasible samples",
            "(b) reconstruction of the joint feasible set",
        ],
        title="CSTR Cascade Linear Simultaneous",
        save=str(Path(__file__).resolve().parent / "corner_simultaneous.png"),
    )


if __name__ == "__main__":
    main()
