import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # use on WSL2

PLOTS = Path(__file__).resolve().parent.parent  # .../reactor_cascade/plots
sys.path.insert(0, str(PLOTS))  # make `tools` importable from any cwd

from tools.trellis import trellis_plot

DATA = PLOTS.parent / "data"
df_live_ = pd.read_csv(DATA / "global_live.csv")
df_dead_ = pd.read_csv(DATA / "global_dead.csv")
df_disc_ = pd.read_csv(DATA / "global_discard.csv")


def plot_df(df_live: pd.DataFrame, df_dead: pd.DataFrame):
    T1_live, tau1_live, T2_live, tau2_live = (
        df_live["T1"],
        df_live["tau1"],
        df_live["T2"],
        df_live["tau2"],
    )
    # crit_live = df_live["crit"]

    T1_dead, tau1_dead, T2_dead, tau2_dead = (
        df_dead["T1"],
        df_dead["tau1"],
        df_dead["T2"],
        df_dead["tau2"],
    )

    trellis_plot(
        [T1_dead, T1_live],
        [T2_dead, T2_live],
        [tau1_dead, tau1_live],
        [tau2_dead, tau2_live],
        x_label="$T_1$ [K]",
        y_label=r"$T_2$ [K]",
        col_label=r"$\tau_1$ [min]",
        row_label=r"$\tau_2$ [min]",
        save=str(Path(__file__).resolve().parent / "trellis_global.png"),
        colors=["crimson", "royalblue"],
        ncols=3,
        nrows=3,
        labels=["dead", "live"],
        # color_by=crit_live, # only usable on one set (e.g. live points only)
    )


def main():
    plot_df(df_live_, df_dead_)


if __name__ == "__main__":
    main()
