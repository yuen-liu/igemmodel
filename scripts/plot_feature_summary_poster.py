"""Poster figures for the per-feature evidence, from feature_summary.csv.

  1. Scatter: energy effect vs controlled steering excess, one point per
     feature. Puts both arms of the SAE analysis in one panel -- which
     features are real on each axis, and which are simply underpowered.
  2. Table: the same rows as a typeset reference block.

Run scripts/build_feature_summary.py first.

Usage:
    python scripts/plot_feature_summary_poster.py
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poster_style import (  # noqa: E402
    COLUMBIA,
    LABEL_SIZE,
    LEGEND_SIZE,
    NAVY,
    NOISE_GRAY,
    TICK_SIZE,
    strip_spines,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_CSV = os.path.join(REPO_ROOT, "sae/results/run4/feature_summary.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")

VERDICT_COLOR = {
    "energy + causal steering": NAVY,
    "energy only": COLUMBIA,
    "steering only": COLUMBIA,
    "no detected effect": NOISE_GRAY,
}
# Circle for a presence effect, triangle for an activation effect.
TERM_MARKER = {"presence": "o", "activation": "^"}

# feature_summary.csv's labels are too long to sit on a scatter point, so the
# plot carries a condensed form of the same thing. Keep these in sync with
# SHORT_LABELS in build_feature_summary.py.
MOTIF = {
    233: "after LSE/LSEE motif",
    1707: "no clear pattern",
    2214: "glycoside hydrolase",
    4657: "Ser after acidic",
    6073: "Leu after acidic",
    6869: "Gly–Lys; P-loop",
    10586: "hydrophobic; iGluR",
    11326: "basic (R/K) + Pro",
    12588: "MUN domain",
    12918: "Phe; tubulin GTPase",
    14247: "hydrophobic I/L/F",
}

# Hand-placed so the two-line callouts do not collide: (dx, dy, ha) in points,
# positioning the feature id; the motif sits one line below it. The point
# positions are fixed by the analysis, so these are stable -- but re-check them
# if the underlying numbers ever move.
LABEL_OFFSET = {
    12588: (0, 30, "center"),
    6073: (0, 30, "center"),
    233: (-10, 30, "right"),
    4657: (16, 6, "left"),
    10586: (-16, 6, "right"),
    12918: (16, 10, "left"),
    14247: (16, 10, "left"),
    11326: (-16, 10, "right"),
    6869: (0, 30, "center"),
    2214: (0, -18, "center"),
    1707: (-16, 10, "right"),
}


def load():
    if not os.path.exists(SUMMARY_CSV):
        raise SystemExit(
            f"{SUMMARY_CSV} not found -- run scripts/build_feature_summary.py first."
        )
    return pd.read_csv(SUMMARY_CSV)


def plot_scatter(df, out_dir, stem, dpi):
    fig, ax = plt.subplots(figsize=(14.5, 9))

    ax.axhline(0, color="#999999", linewidth=1.1, linestyle="--", zorder=1)
    ax.axvline(0, color="#999999", linewidth=1.1, linestyle="--", zorder=1)

    # Area scales with the number of paired steering tests, so a null with
    # 60 pairs cannot be mistaken for a null with 2,865.
    sizes = 60 + 340 * np.sqrt(df["steer_n_pairs"] / df["steer_n_pairs"].max())

    for (term, verdict), g in df.groupby(["term", "verdict"]):
        ax.scatter(
            g["energy_coef"], g["steer_excess_pp"],
            s=sizes.loc[g.index],
            marker=TERM_MARKER[term],
            facecolors=VERDICT_COLOR[verdict],
            edgecolors="white", linewidths=1.6,
            alpha=0.95, zorder=3,
        )

    for _, row in df.iterrows():
        fid = int(row["feature"])
        big = row["verdict"] != "no detected effect"
        dx, dy, ha = LABEL_OFFSET[fid]
        xy = (row["energy_coef"], row["steer_excess_pp"])

        ax.annotate(
            str(fid), xy,
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="center",
            fontsize=LEGEND_SIZE if big else LEGEND_SIZE - 3,
            fontweight="bold" if big else "normal",
            color="#222222" if big else "#666666",
        )
        ax.annotate(
            MOTIF[fid], xy,
            textcoords="offset points", xytext=(dx, dy - 16),
            ha=ha, va="center",
            fontsize=LEGEND_SIZE - 4,
            style="italic",
            color="#444444" if big else "#8A8A8A",
        )

    ax.set_xlabel(
        "Energy effect (kcal/mol)          ←  more favorable binding",
        fontsize=LABEL_SIZE,
    )
    ax.set_ylabel("Steering excess vs. random (pp)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.2, length=6)
    strip_spines(ax)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=NAVY,
                   markersize=15, markeredgecolor="white",
                   label="energy + causal steering"),
        plt.Line2D([], [], marker="o", linestyle="none", color=COLUMBIA,
                   markersize=15, markeredgecolor="white", label="energy only"),
        plt.Line2D([], [], marker="o", linestyle="none", color=NOISE_GRAY,
                   markersize=15, markeredgecolor="white", label="no detected effect"),
        plt.Line2D([], [], marker="o", linestyle="none", color="#555555",
                   markersize=13, label="presence effect"),
        plt.Line2D([], [], marker="^", linestyle="none", color="#555555",
                   markersize=13, label="activation effect"),
        plt.Line2D([], [], marker="o", linestyle="none", color="#BBBBBB",
                   markersize=7, label="marker area ∝ n steering pairs"),
    ]
    # The callout labels occupy the interior, so the legend sits outside to
    # the right rather than covering points.
    ax.legend(handles=handles, fontsize=LEGEND_SIZE - 1, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              labelspacing=0.8, handletextpad=0.7)

    fig.tight_layout()
    save(fig, out_dir, stem, dpi)


def plot_table(df, out_dir, stem, dpi):
    df = df.sort_values(["energy_q", "steer_p"])
    fig, ax = plt.subplots(figsize=(18.5, 7.5))
    ax.axis("off")

    cols = [
        ("Feature", 0.000, "left"),
        ("Fires on", 0.068, "left"),
        ("Energy", 0.400, "left"),
        ("Steering vs. random", 0.575, "left"),
        ("Verdict", 0.800, "left"),
    ]
    n = len(df)
    top, row_h = 0.90, 0.078

    for name, x, ha in cols:
        ax.text(x, top, name, fontsize=LEGEND_SIZE, fontweight="bold",
                ha=ha, va="bottom", color="#222222", transform=ax.transAxes)
    ax.plot([0, 1], [top - 0.018] * 2, color="#222222", linewidth=1.6,
            transform=ax.transAxes, clip_on=False)

    for i, (_, r) in enumerate(df.iterrows()):
        y = top - 0.055 - i * row_h
        color = VERDICT_COLOR[r["verdict"]]
        strong = r["verdict"] != "no detected effect"
        weight = "bold" if strong else "normal"
        txt = "#222222" if strong else "#6A6A6A"

        if i % 2 == 1:
            ax.add_patch(plt.Rectangle(
                (0, y - 0.022), 1, row_h * 0.82, transform=ax.transAxes,
                facecolor="#F4F6F8", edgecolor="none", zorder=0))

        ax.text(cols[0][1], y, str(int(r["feature"])), fontsize=LEGEND_SIZE,
                fontweight="bold", color=color, va="center", transform=ax.transAxes)
        ax.text(cols[1][1], y, r["label"], fontsize=LEGEND_SIZE - 2,
                color=txt, va="center", transform=ax.transAxes)

        q = r["energy_q"]
        qs = "q<0.001" if q < 0.001 else f"q={q:.3f}"
        term_abbr = "act" if r["term"] == "activation" else "pres"
        ax.text(cols[2][1], y,
                f"{r['energy_coef']:+.1f}  ({term_abbr}, {qs})".replace("-", "\u2212"),
                fontsize=LEGEND_SIZE - 2, color=txt, va="center",
                fontweight=weight, transform=ax.transAxes)

        p = r["steer_p"]
        ps = "p<0.001" if p < 0.001 else f"p={p:.2f}"
        ax.text(cols[3][1], y,
                f"{r['steer_excess_pp']:+.1f}pp  ({ps}, n={int(r['steer_n_pairs'])})".replace("-", "\u2212"),
                fontsize=LEGEND_SIZE - 2, color=txt, va="center",
                fontweight=weight, transform=ax.transAxes)
        ax.text(cols[4][1], y, r["verdict"], fontsize=LEGEND_SIZE - 2,
                color=color, fontweight=weight, va="center", transform=ax.transAxes)

    ax.text(0, top - 0.055 - n * row_h - 0.02,
            "Energy: confound-controlled regression, n=477 designs, BH-FDR. "
            "Steering: paired vs. norm-matched random, ≤2×, "
            "Bonferroni threshold p<0.0045.",
            fontsize=LEGEND_SIZE - 4, color="#6A6A6A", va="top",
            transform=ax.transAxes)

    fig.tight_layout()
    save(fig, out_dir, stem, dpi)


def save(fig, out_dir, stem, dpi):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    df = load()
    plot_scatter(df, args.out_dir, "vilip1_feature_energy_vs_steering_poster", args.dpi)
    plot_table(df, args.out_dir, "vilip1_feature_summary_table_poster", args.dpi)


if __name__ == "__main__":
    main()
