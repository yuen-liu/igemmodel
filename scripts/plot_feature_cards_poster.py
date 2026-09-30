"""Poster figure: one card per headline feature.

Shows what each feature actually fires on -- real activating contexts from
feature_top_examples.csv, the bracketed residue being the one that activates
it -- next to the evidence for it on each arm.

Deliberately NOT a sequence logo: only the top 15 examples per feature were
saved, and they are the top 15 *by activation*, so a logo built from them
would show far more conservation than the feature really has. Verbatim
examples are honest as examples; a logo would be a distribution claim the
data cannot support.

Usage:
    python scripts/plot_feature_cards_poster.py
    python scripts/plot_feature_cards_poster.py --features 233,6073
"""

import argparse
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poster_style import COLUMBIA, LEGEND_SIZE, NAVY, NOISE_GRAY  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN4 = os.path.join(REPO_ROOT, "sae/results/run4")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")

N_CONTEXTS = 5

# Added to every font size on the card. Bumping text without widening the card
# will overrun the borders, so main() widens per bump to match.
BUMP = 0


def fs(delta=0):
    return LEGEND_SIZE + delta + BUMP


# "n candidate designs (of 64,998)" as reported in sae/RESULTS.md's feature
# table. Recomputing these needs the 210 MB interface_features_combined.yaml.
N_DESIGNS = {233: "46,327", 6073: "273", 11326: "64,997", 14247: "46,995",
             10586: "41", 4657: "728", 1707: "50,765", 12588: "18",
             12918: "97", 2214: "19", 6869: "74"}

HEADLINE = {
    233: "Basic/acidic after an LSE/LSEE motif",
    6073: "Leu preceded by acidic (D/E); 6 of top 12",
    11326: "Basic (R/K) + Pro, before acidic/Pro-rich",
}

VERDICT_COLOR = {
    "energy + causal steering": NAVY,
    "energy only": COLUMBIA,
    "steering only": COLUMBIA,
    "no detected effect": NOISE_GRAY,
}


def card(ax, feat, summary, contexts):
    row = summary[summary.feature == feat].iloc[0]
    color = VERDICT_COLOR[row["verdict"]]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#FBFCFD", edgecolor="#D8DEE4", linewidth=1.4,
        transform=ax.transAxes, zorder=0))
    # Colour bar keys the card to the scatter's verdict colours.
    ax.add_patch(plt.Rectangle(
        (0.02, 0.905), 0.96, 0.075, facecolor=color, edgecolor="none",
        transform=ax.transAxes, zorder=1))

    ax.text(0.06, 0.9425, f"Feature {feat}", fontsize=fs(1),
            fontweight="bold", color="white", va="center", zorder=2)
    # Verdict on its own line below the bar. Right-aligning it beside the
    # feature name collides once the card is narrow enough for the poster.
    ax.text(0.06, 0.862, row["verdict"], fontsize=fs(-5),
            color=color, fontweight="bold", va="center")

    # Wrapped, not shrunk: at the narrow card width these labels overrun the
    # right border on one line, and shrinking them further costs legibility.
    for i, line in enumerate(textwrap.wrap(HEADLINE[feat], width=30)[:2]):
        ax.text(0.06, 0.805 - i * 0.045, line, fontsize=fs(-4),
                style="italic", color="#333333", va="center")

    # The right-hand column is the SAE activation at the bracketed residue;
    # label it, or the numbers read as unexplained.
    ax.text(0.06, 0.700, "Top activating sites", fontsize=fs(-6),
            color="#7A7A7A", va="center")
    ax.text(0.94, 0.700, "activation", fontsize=fs(-6),
            color="#7A7A7A", ha="right", va="center")
    y = 0.640
    for _, c in contexts.iterrows():
        ax.text(0.06, y, c["context"], fontsize=fs(-4),
                family="monospace", color="#1A1A1A", va="center")
        ax.text(0.94, y, f"{c['activation']:.1f}", fontsize=fs(-6),
                family="monospace", color="#8A8A8A", ha="right", va="center")
        y -= 0.056

    ax.plot([0.06, 0.94], [y + 0.020] * 2, color="#E2E7EC", linewidth=1.2)

    q = row["energy_q"]
    p = row["steer_p"]
    term = "activation" if row["term"] == "activation" else "presence"
    stats = [
        ("Fires in", f"{N_DESIGNS.get(feat, '?')} / 64,998 designs"),
        (f"Energy ({term})",
         f"{row['energy_coef']:+.1f} kcal/mol   q={q:.3f}".replace("-", "−")),
        ("Steering vs random",
         f"{row['steer_excess_pp']:+.1f}pp   "
         f"{'p<0.001' if p < 0.001 else f'p={p:.2f}'}   n={int(row['steer_n_pairs'])}"
         .replace("-", "−")),
    ]
    # Label stacked above value rather than beside it, and the p/n split off the
    # value line: side-by-side made the card as wide as its longest string.
    y -= 0.060
    for k, v in stats:
        ax.text(0.06, y, k, fontsize=fs(-6), color="#7A7A7A", va="center")
        ax.text(0.06, y - 0.045, v, fontsize=fs(-4), color="#1A1A1A",
                va="center")
        y -= 0.098


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="233,6073,11326")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--stem", default="vilip1_feature_cards_poster")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--font-bump", type=int, default=2,
                    help="Add N points to every font size; the card widens to match.")
    args = ap.parse_args()

    global BUMP
    BUMP = args.font_bump

    feats = [int(f) for f in args.features.split(",")]
    summary = pd.read_csv(os.path.join(RUN4, "feature_summary.csv"))
    examples = pd.read_csv(os.path.join(RUN4, "feature_top_examples.csv"))

    fig, axes = plt.subplots(1, len(feats), figsize=(4.3 * (1 + 0.05 * BUMP) * len(feats), 6.6 * (1 + 0.02 * BUMP)))
    if len(feats) == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        ctx = examples[examples.feature == feat].nsmallest(N_CONTEXTS, "rank")
        card(ax, feat, summary, ctx)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"{args.stem}.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
