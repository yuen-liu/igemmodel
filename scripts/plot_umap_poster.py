"""Poster figure: 2D UMAP of ESM-C layer 18 embeddings, HDBSCAN clusters,
with the wet-lab representatives starred.

Reads the artifacts that
notebooks/bridget/esmc_embedding_analysis/esmc_deepdive_vilip1_composite_hotspot_combined_layer18.ipynb
already wrote -- it does not recompute UMAP or HDBSCAN, so the figure is
guaranteed to match the coordinates and picks the notebook reported:

  umap_coordinates.csv        -- protein, UMAP1, UMAP2, UMAP3, cluster
  dunbrack_top5_per_cluster.csv -- Selection 1, top 5 per cluster by ipsae/iptm/ipae

Usage:
    python scripts/plot_umap_poster.py
    python scripts/plot_umap_poster.py --no-noise-reps --dpi 600
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poster_style import (  # noqa: E402
    ACCENTS,
    COLUMBIA,
    LABEL_SIZE,
    LEGEND_SIZE,
    NAVY,
    NOISE_GRAY,
    TICK_SIZE,
)

# Clusters 1/2/3 are 4/5/7 points each. They are real clusters that contribute
# 14 of the 29 representatives, so they get their own ACCENTS color rather than
# being folded into a neighbor.
#
# Assigned per cluster id below; anything unlisted falls through to ACCENTS.
CLUSTER_COLORS = {0: NAVY, 4: COLUMBIA}

# Clusters smaller than this are drawn as emphasized points and get the zoom inset.
MICRO_CLUSTER_MAX = 50

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLUSTER_DIR = os.path.join(
    REPO_ROOT,
    "data/vilip1-design-composite_hotspot-combined/clustering/layer_18",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster-dir", default=DEFAULT_CLUSTER_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--stem", default="vilip1_layer18_umap_poster")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument(
        "--no-inset",
        action="store_true",
        help="Drop the zoom panel on the micro-clusters.",
    )
    p.add_argument(
        "--no-noise-reps",
        action="store_true",
        help="Drop the 5 representatives picked from the noise group, so only "
        "picks from real clusters get stars.",
    )
    return p


def assign_colors(umap_df):
    """Map each real cluster id to a color, biggest cluster first."""
    ids = sorted(
        (c for c in umap_df["cluster"].unique() if c != -1),
        key=lambda c: -(umap_df["cluster"] == c).sum(),
    )
    colors, accent = {}, 0
    for cid in ids:
        if cid in CLUSTER_COLORS:
            colors[cid] = CLUSTER_COLORS[cid]
        else:
            colors[cid] = ACCENTS[accent % len(ACCENTS)]
            accent += 1
    return colors


def add_inset(ax, umap_df, draw_clusters):
    """Zoom on the micro-clusters.

    Clusters 1/2/3 hold 16 designs between them but contribute 14 of the 29
    representatives, and at full extent they are a single speck buried under
    their own stars. Without this panel the figure shows half the selection
    as one dot.
    """
    micro = umap_df[
        umap_df["cluster"].map(lambda c: c != -1 and (umap_df["cluster"] == c).sum() < MICRO_CLUSTER_MAX)
    ]
    if micro.empty:
        return

    pad_x = max(0.18, (micro["UMAP1"].max() - micro["UMAP1"].min()) * 0.35)
    pad_y = max(0.18, (micro["UMAP2"].max() - micro["UMAP2"].min()) * 0.35)
    x0, x1 = micro["UMAP1"].min() - pad_x, micro["UMAP1"].max() + pad_x
    y0, y1 = micro["UMAP2"].min() - pad_y, micro["UMAP2"].max() + pad_y

    axins = ax.inset_axes(
        [0.60, 0.62, 0.38, 0.34],
        xlim=(x0, x1),
        ylim=(y0, y1),
        xticks=[],
        yticks=[],
    )
    draw_clusters(axins, with_labels=False, zoom=True)
    for side in ("top", "right", "bottom", "left"):
        axins.spines[side].set_linewidth(1.2)
        axins.spines[side].set_color("#444444")
    axins.set_facecolor("white")
    axins.set_title(
        "Clusters 1–3 (zoom)", fontsize=LEGEND_SIZE - 2, color="#444444", pad=6
    )
    ax.indicate_inset_zoom(axins, edgecolor="#444444", linewidth=1.2, alpha=0.9)


def main():
    args = build_parser().parse_args()

    umap_df = pd.read_csv(os.path.join(args.cluster_dir, "umap_coordinates.csv"))
    picks = pd.read_csv(os.path.join(args.cluster_dir, "dunbrack_top5_per_cluster.csv"))

    if args.no_noise_reps:
        picks = picks[picks["cluster"] != -1]

    # Coordinates live in umap_coordinates.csv keyed by "protein"; the picks
    # table calls the same column "id".
    rep_coords = picks.merge(
        umap_df[["protein", "UMAP1", "UMAP2"]],
        left_on="id",
        right_on="protein",
        how="inner",
    )
    if len(rep_coords) != len(picks):
        missing = set(picks["id"]) - set(rep_coords["id"])
        raise SystemExit(
            f"{len(missing)} representative(s) have no UMAP coordinates: "
            f"{sorted(missing)[:5]}"
        )

    fig, ax = plt.subplots(figsize=(11, 8.5))

    color_by_cluster = assign_colors(umap_df)
    noise = umap_df[umap_df["cluster"] == -1]

    def draw_clusters(target, with_labels, zoom=False):
        """Paint the population onto `target` (main axes or the zoom inset).

        In the inset the micro-cluster dots are enlarged and the stars shrunk,
        so each selected design still shows the color of the cluster it was
        drawn from instead of vanishing under its own marker.
        """
        big_s, micro_s, noise_s, star_s = (18, 260, 26, 130) if zoom else (11, 110, 9, 260)

        for cid in sorted(color_by_cluster, key=lambda c: -(umap_df["cluster"] == c).sum()):
            pts = umap_df[umap_df["cluster"] == cid]
            small = len(pts) < MICRO_CLUSTER_MAX
            target.scatter(
                pts["UMAP1"],
                pts["UMAP2"],
                c=color_by_cluster[cid],
                s=micro_s if small else big_s,
                alpha=1.0 if small else 0.8,
                linewidths=1.0 if small else 0,
                edgecolors="white" if small else "none",
                label=f"Cluster {cid}  n={len(pts)}" if with_labels else None,
                zorder=4 if small else 2,
            )

        # Noise goes ON TOP of the big clusters -- the speckle inside the blobs
        # is exactly what the legend needs to account for, so it cannot hide
        # behind the fill. Still under the micro-clusters and the stars.
        target.scatter(
            noise["UMAP1"],
            noise["UMAP2"],
            c=NOISE_GRAY,
            s=noise_s,
            alpha=0.75,
            linewidths=0,
            label=f"Unclustered (HDBSCAN −1)  n={len(noise)}" if with_labels else None,
            zorder=3,
        )

        target.scatter(
            rep_coords["UMAP1"],
            rep_coords["UMAP2"],
            marker="*",
            s=star_s,
            c="black",
            edgecolors="white",
            linewidths=0.9,
            label=f"Selected for wet lab  n={len(rep_coords)}" if with_labels else None,
            zorder=5,
        )

    draw_clusters(ax, with_labels=True)

    ax.set_xlabel("UMAP 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2, length=6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if not args.no_inset:
        add_inset(ax, umap_df, draw_clusters)

    # The upper-right quadrant of the data range is empty, so the legend sits
    # inside the axes rather than stealing width from the scatter.
    legend = ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.30, 0.34) if not args.no_inset else (0.30, 0.62),
        markerscale=1.3,
        labelspacing=0.75,
        handletextpad=0.5,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"{args.stem}.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
