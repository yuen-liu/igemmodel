"""Poster figure: per-feature effect on interface interaction energy.

Forest plot of the confound-controlled regression documented in
sae/RESULTS.md ("Correction (2026-09-27)"): dE_interaction regressed on
binder length + every feature's presence AND activation strength at once,
HC3 robust SEs, BH-FDR across feature terms.

The fit is a verbatim copy of sae/07_energy/feature_energy_regression.py
(which prints but saves nothing). Because it is a copy, main() asserts the
three headline coefficients still match the values RESULTS.md quotes --
if the analysis script or its inputs change, this figure fails loudly
rather than drifting away from the text.

Usage:
    python scripts/plot_energy_regression_poster.py
    python scripts/plot_energy_regression_poster.py --dpi 600
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

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
ENERGY_CSV = os.path.join(REPO_ROOT, "sae/results/run4/energy_profile_combined.csv")
SAMPLE_YAML = os.path.join(
    REPO_ROOT, "sae/results/run4/yamls/interface_features_energy_sample.yaml"
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")

# Quoted in sae/RESULTS.md; the fit below must still reproduce them.
EXPECTED = {"act_233_perSD": -13.5, "act_11326_perSD": -12.9, "present_6073": -30.1}

# RESULTS.md: "233's and 14247's *presence* coefficients are unreliable --
# only ~30 designs lack either feature, mostly the same 30, r=0.92 between
# their absence, so the model can't separate them there." They are the two
# largest coefficients in the table, so they are drawn but flagged.
UNRELIABLE = {"present_233", "present_14247"}

# Significance tiers, by BH-FDR q. RESULTS.md groups these as "holds up
# robustly (q<=0.01)" vs "borderline (q~0.06)", but the robust group's actual
# q is 0.0114 for act_233/present_6073 (only act_11326 is truly <=0.01), so
# the cut is drawn at 0.05 to reproduce RESULTS.md's grouping honestly.
Q_STRONG = 0.05
Q_BORDERLINE = 0.10

XLIM = (-90, 70)


def fit():
    """Verbatim from sae/07_energy/feature_energy_regression.py."""
    df = pd.read_csv(ENERGY_CSV)
    y_ = yaml.load(open(SAMPLE_YAML), Loader=yaml.SafeLoader)
    act = df.pivot_table(index="design_id", columns="feature_id", values="max_activation")
    dE = df.drop_duplicates("design_id").set_index("design_id").loc[act.index, "dE_interaction"]
    nres = pd.Series(
        [float(y_["designs"][d]["n_binder_residues"]) for d in act.index], index=act.index
    )

    X = {"binder_len_per10res": (nres - nres.mean()) / 10}
    for f in act.columns:
        present = act[f].notna()
        if present.mean() < 1:
            X[f"present_{f}"] = present.astype(float)
        a = act[f][present]
        z = pd.Series(0.0, index=act.index)
        if a.std() > 0:
            z[present] = (a - a.mean()) / a.std()
        X[f"act_{f}_perSD"] = z
    X = pd.DataFrame(X, index=act.index)

    Xm = np.column_stack([np.ones(len(X)), X.values])
    yv = dE.values
    XtXi = np.linalg.pinv(Xm.T @ Xm)
    beta = XtXi @ Xm.T @ yv
    e = yv - Xm @ beta
    h = np.einsum("ij,jk,ik->i", Xm, XtXi, Xm)
    meat = Xm.T @ (Xm * (e**2 / (1 - h) ** 2)[:, None])
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    dof = len(yv) - Xm.shape[1]
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - (e**2).sum() / ((yv - yv.mean()) ** 2).sum()

    res = pd.DataFrame(
        {"coef": beta, "se": se, "p": p}, index=["intercept"] + list(X.columns)
    )
    feat = res.index.str.startswith(("present_", "act_"))
    res["q"] = np.nan
    res.loc[feat, "q"] = stats.false_discovery_control(res.loc[feat, "p"].values)
    return res[feat].copy(), len(yv), r2


def pretty(term):
    if term.startswith("act_"):
        return f"{term[4:].split('_')[0]}  · activation"
    return f"{term[len('present_'):]}  · presence"


def color_for(term, q):
    if term in UNRELIABLE:
        return NOISE_GRAY
    if q <= Q_STRONG:
        return NAVY
    if q <= Q_BORDERLINE:
        return COLUMBIA
    return NOISE_GRAY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--stem", default="vilip1_feature_energy_regression_poster")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    res, n_designs, r2 = fit()

    for term, want in EXPECTED.items():
        got = res.loc[term, "coef"]
        if abs(got - want) > 0.05:
            raise SystemExit(
                f"{term} = {got:.2f}, but RESULTS.md quotes {want}. The fit no "
                "longer matches the documented analysis -- reconcile before plotting."
            )

    res["lo"] = res["coef"] - 1.96 * res["se"]
    res["hi"] = res["coef"] + 1.96 * res["se"]
    res = res.sort_values("coef", ascending=False)  # most favorable ends up on top

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.axvline(0, color="#333333", linewidth=1.3, linestyle="--", zorder=1, alpha=0.8)

    for y, (term, row) in enumerate(res.iterrows()):
        color = color_for(term, row["q"])
        lo, hi = max(row["lo"], XLIM[0]), min(row["hi"], XLIM[1])
        ax.plot([lo, hi], [y, y], color=color, linewidth=3.0, solid_capstyle="round", zorder=2)

        # Arrow caps where the interval runs off the axis.
        if row["lo"] < XLIM[0]:
            ax.plot(XLIM[0], y, marker="<", color=color, markersize=11, zorder=3)
        if row["hi"] > XLIM[1]:
            ax.plot(XLIM[1], y, marker=">", color=color, markersize=11, zorder=3)

        open_marker = term in UNRELIABLE
        if XLIM[0] <= row["coef"] <= XLIM[1]:
            ax.scatter(
                row["coef"], y,
                s=190, zorder=4,
                facecolors="white" if open_marker else color,
                edgecolors=color, linewidths=2.2,
            )
        else:
            # Point estimate itself is off-scale: park it at the edge and
            # print the real value so nothing is silently hidden.
            edge = XLIM[0] if row["coef"] < XLIM[0] else XLIM[1]
            ax.scatter(edge, y, s=190, zorder=4, facecolors="white",
                       edgecolors=color, linewidths=2.2)
            ax.annotate(
                f"{row['coef']:+.0f}".replace("-", "−"),
                (edge, y),
                textcoords="offset points",
                # Lifted off the row so it does not sit on the interval line.
                xytext=(26 if edge == XLIM[0] else -26, 13),
                ha="left" if edge == XLIM[0] else "right",
                va="center", fontsize=LEGEND_SIZE - 3, color="#333333",
            )

    labels = [pretty(t) + ("  †" if t in UNRELIABLE else "") for t in res.index]
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(labels, fontsize=TICK_SIZE - 3)
    ax.set_ylim(-0.8, len(res) - 0.2)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Effect on interaction energy (kcal/mol)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE, width=1.2, length=6)
    ax.tick_params(axis="y", length=0)

    strip_spines(ax)

    ax.annotate(
        "←  more favorable binding",
        xy=(0.02, -0.105), xycoords="axes fraction",
        fontsize=LEGEND_SIZE, color="#333333",
    )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="-", linewidth=3, color=NAVY,
                   markersize=13, label="q ≤ 0.05"),
        plt.Line2D([], [], marker="o", linestyle="-", linewidth=3, color=COLUMBIA,
                   markersize=13, label="q ≤ 0.10"),
        plt.Line2D([], [], marker="o", linestyle="-", linewidth=3, color=NOISE_GRAY,
                   markersize=13, label="not significant"),
        plt.Line2D([], [], marker="o", linestyle="none", color=NOISE_GRAY,
                   markerfacecolor="white", markeredgewidth=2.2, markersize=13,
                   label="† collinear, uninterpretable"),
    ]
    # Upper right: the top rows' intervals all end well left of zero, so the
    # legend lands on empty canvas there.
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, frameon=False,
              loc="upper right", labelspacing=0.7, handletextpad=0.6,
              borderaxespad=1.0)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"{args.stem}.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    print(f"n={n_designs} designs, R2={r2:.3f}  (for the caption)")
    plt.close(fig)


if __name__ == "__main__":
    main()
