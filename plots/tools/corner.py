"""
General corner-plot template for D-dimensional point clouds.

Lower-triangle pairwise scatter: for D dimensions the grid is
(D-1) x (D-1) panels, where panel (r, c) shows dimension c on x against
dimension r+1 on y (only c <= r is drawn; the upper triangle stays
empty). Each panel carries the design box as a dashed rectangle, with
axis ticks at the box bounds.

Point sets are passed as a list of dicts and drawn in list order
(painter's algorithm: last set sits on top). A set's `data` is either

  * an (N, D) array — drawn in every panel using columns (c, r+1), or
  * a dict {(i, j): (N, 2) array} — drawn only in the panel whose x-dim
    is i and y-dim is j (requires i < j), e.g. unit-local samples that
    only exist for same-unit dimension pairs.

Optionally the whole corner is repeated as side-by-side BLOCKS (an
(a)/(b) comparison): pass nblocks > 1 and give each set a `blocks` list
saying which blocks it appears in (default: all).

Usage:

    from corner import corner_plot
    fig = corner_plot(
        [
            dict(data=init, color="0.75"),                    # everywhere
            dict(data={(0, 1): u1, (2, 3): u2},               # same-unit pairs
                 color="red", alpha=0.4),
            dict(data=recon, color="blue", alpha=0.4,
                 blocks=[1]),                                 # block (b) only
        ],
        labels=["$T_1$ [K]", r"$\tau_1$ [min]", "$T_2$ [K]", r"$\tau_2$ [min]"],
        bounds=[(298.15, 353.15), (2, 30), (298.15, 353.15), (2, 30)],
        nblocks=2,
        captions=["(a) unit-local feasible samples",
                  "(b) reconstruction of the joint feasible set"],
        title="CSTR Cascade Linear Simultaneous",
        save="corner_simultaneous.png",
    )
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

__all__ = ["corner_plot"]

# fixed set order: first set, second set, ... (never cycled)
DEFAULT_COLORS = ["0.75", "crimson", "royalblue", "seagreen", "darkorange"]


def _norm_set(spec, k, ndim, nblocks):
    """Fill in a point-set spec's defaults and validate its data shape."""
    if "data" not in spec:
        raise ValueError(f"point set {k}: missing 'data'")
    data = spec["data"]
    if isinstance(data, dict):
        norm = {}
        for pair, pts in data.items():
            i, j = pair
            if not (0 <= i < j < ndim):
                raise ValueError(
                    f"point set {k}: pair {pair} invalid, need 0 <= i < j < {ndim}"
                )
            pts = np.asarray(pts, float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(
                    f"point set {k}, pair {pair}: expected (N, 2) array, "
                    f"got shape {pts.shape}"
                )
            norm[(i, j)] = pts
        data = norm
    else:
        data = np.asarray(data, float)
        if data.ndim != 2 or data.shape[1] != ndim:
            raise ValueError(
                f"point set {k}: expected (N, {ndim}) array, got shape {data.shape}"
            )
    blocks = spec.get("blocks")
    blocks = set(range(nblocks)) if blocks is None else set(blocks)
    if not blocks <= set(range(nblocks)):
        raise ValueError(
            f"point set {k}: blocks {sorted(blocks)} outside 0..{nblocks - 1}"
        )
    return {
        "data": data,
        "color": spec.get("color", DEFAULT_COLORS[k % len(DEFAULT_COLORS)]),
        "ms": spec.get("ms", 2),
        "alpha": spec.get("alpha", None),
        "label": spec.get("label", None),
        "blocks": blocks,
    }


def corner_plot(
    sets,
    *,
    labels,
    bounds,
    nblocks=1,
    captions=None,
    title=None,
    pad=0.06,
    panel_size=3.0,
    spacer=0.25,
    legend=False,
    save=None,
    dpi=300,
    verbose=True,
):
    """Lower-triangle corner plot of D-dimensional point sets, optionally
    repeated as several side-by-side blocks.

    Parameters
    ----------
    sets : list of dict
        Point sets, drawn in order (last on top). Keys per set:
          data   : (N, D) array, or dict {(i, j): (N, 2)} for per-pair
                   points shown only in panel x-dim i vs y-dim j (i < j).
          color  : matplotlib color (default: grey, red, blue, ...).
          ms     : marker size in points (default 2).
          alpha  : marker alpha (default opaque).
          label  : legend entry (used when legend=True).
          blocks : iterable of block indices this set appears in
                   (default: all blocks).
    labels : list of str, length D
        Axis label per dimension (LaTeX ok, e.g. r"$\\tau_1$ [min]").
    bounds : list of (lo, hi), length D
        Design box per dimension; drawn as a dashed rectangle, ticks sit
        at these bounds, panel limits extend `pad` beyond them.
    nblocks : int
        Number of side-by-side corner blocks (1 = plain corner plot).
    captions : list of str, length nblocks, or None
        Text centered under each block (e.g. "(a) ...", "(b) ...").
    title : str or None
        Figure title.
    pad : float
        Fraction of the bound range added around the box on each side.
    panel_size : float
        Edge length of one panel in inches.
    spacer : float
        Width of the gap column between blocks, in panel widths.
    legend : bool
        Add a figure legend from the sets' `label` entries.
    save : str or None
        If given, save the figure to this path (bbox_inches="tight").

    Returns
    -------
    fig : matplotlib Figure
    """
    ndim = len(labels)
    if len(bounds) != ndim:
        raise ValueError(f"labels has {ndim} entries but bounds has {len(bounds)}")
    if ndim < 2:
        raise ValueError("need at least 2 dimensions")
    if captions is not None and len(captions) != nblocks:
        raise ValueError(f"need {nblocks} captions, got {len(captions)}")
    sets = [_norm_set(s, k, ndim, nblocks) for k, s in enumerate(sets)]

    npan = ndim - 1  # rows and columns per corner block
    # blocks separated by hidden spacer columns
    width_ratios = [1.0] * npan + ([spacer] + [1.0] * npan) * (nblocks - 1)
    fig, axs = plt.subplots(
        npan,
        len(width_ratios),
        figsize=(panel_size * sum(width_ratios), panel_size * npan),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    for b in range(1, nblocks):  # spacer columns stay empty
        for ax in axs[:, b * (npan + 1) - 1]:
            ax.set_visible(False)

    for b in range(nblocks):
        axb = axs[:, b * (npan + 1) : b * (npan + 1) + npan]
        for r in range(npan):
            for c in range(npan):
                ax = axb[r, c]
                if c > r:  # upper triangle stays empty
                    ax.set_visible(False)
                    continue
                xi, yi = c, r + 1  # panel (r, c): dim c on x vs dim r+1 on y
                (xlo, xhi), (ylo, yhi) = bounds[xi], bounds[yi]
                for s in sets:  # painter's order: list order, last on top
                    if b not in s["blocks"]:
                        continue
                    d = s["data"]
                    if isinstance(d, dict):
                        if (xi, yi) not in d:
                            continue
                        xv, yv = d[(xi, yi)][:, 0], d[(xi, yi)][:, 1]
                    else:
                        xv, yv = d[:, xi], d[:, yi]
                    ax.plot(
                        xv, yv, ".", color=s["color"], ms=s["ms"], alpha=s["alpha"]
                    )
                ax.add_patch(
                    Rectangle(
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
                if r == npan - 1:
                    ax.set_xlabel(labels[xi])
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel(labels[yi])
                else:
                    ax.tick_params(labelleft=False)
        fig.align_ylabels(axb[:, 0])  # same x-position regardless of tick labels
        if captions is not None:
            # center the caption under this block's bottom row of panels
            left = axb[npan - 1, 0].get_position()
            right = axb[npan - 1, npan - 1].get_position()
            fig.text((left.x0 + right.x1) / 2, 0.02, captions[b], ha="center")

    if title:
        fig.suptitle(title)

    if legend:
        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker=".",
                markersize=9,
                color=s["color"],
                alpha=s["alpha"],
                label=s["label"],
            )
            for s in sets
            if s["label"] is not None
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.985),
                frameon=False,
                fontsize=9,
            )

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"wrote {save}")
    return fig
