"""Consolidate the per-feature evidence into one table.

RESULTS.md currently splits this across two tables, one of which is
retracted (the uncontrolled `aa_argmax_changed` rates). This builds a
single row per feature carrying both arms, with the *controlled* steering
numbers only, so the retracted rates cannot be read by accident.

Energy arm: coefficient from the confound-controlled regression (see
scripts/plot_energy_regression_poster.py, which reproduces
sae/07_energy/feature_energy_regression.py).

Steering arm: paired feature-vs-norm-matched-random McNemar over the three
<=2x control files, deduplicated on (design, feature, site, alpha) because
the pools overlap. That rule reproduces RESULTS.md's published n and
feature-rate for all 11 features exactly.

Writes sae/results/run4/feature_summary.csv.
"""

import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_energy_regression_poster import UNRELIABLE, fit  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_FILES = [
    "injection_with_controls.csv",
    "injection_with_controls_65kpool.csv",
    "injection_topk_controls.csv",
]
OUT_CSV = os.path.join(REPO_ROOT, "sae/results/run4/feature_summary.csv")

# Bonferroni for the 11 per-feature steering comparisons, per RESULTS.md.
STEER_ALPHA = 0.05 / 11
Q_STRONG = 0.05

# Condensed from feature_labels.csv, matching the short forms RESULTS.md uses.
SHORT_LABELS = {
    233: "Basic/acidic after an LSE/LSEE motif",
    1707: "No clear pattern",
    2214: "Glycoside hydrolase superfamily",
    4657: "Ser after acidic; RTN1-4/CAPS family",
    6073: "Leu preceded by acidic (D/E)",
    6869: "Gly before Lys (G-K); P-loop NTPase/GTPase",
    10586: "Hydrophobic (G/L/F/I/W); iGluR-associated",
    11326: "Basic (R/K) + Pro, before acidic/Pro-rich",
    12588: "MUN domain (IPR010439)",
    12918: "Phe in TGLQG[F]L; tubulin/FtsZ GTPase",
    14247: "Hydrophobic (I/L/F); no dominant domain",
}


def steering_table():
    d = pd.concat(
        [pd.read_csv(os.path.join(REPO_ROOT, f)) for f in CONTROL_FILES],
        ignore_index=True,
    )
    d = d[d["alpha_multiplier"] <= 2.0]
    key = ["design_id", "feature_id", "resnum", "alpha_multiplier"]
    f = d[d.direction_mode == "feature"].drop_duplicates(key).set_index(key)["aa_argmax_changed"]
    r = d[d.direction_mode == "random"].drop_duplicates(key).set_index(key)["aa_argmax_changed"]
    j = pd.concat({"f": f, "r": r}, axis=1).dropna()

    rows = []
    for fid, g in j.groupby(level="feature_id"):
        b = int(((g.f == 1) & (g.r == 0)).sum())
        c = int(((g.f == 0) & (g.r == 1)).sum())
        p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append({
            "feature": fid,
            "steer_n_pairs": len(g),
            "steer_feature_rate": g.f.mean(),
            "steer_random_rate": g.r.mean(),
            "steer_excess_pp": (g.f.mean() - g.r.mean()) * 100,
            "steer_p": p,
        })
    return pd.DataFrame(rows)


def energy_table():
    res, n_designs, r2 = fit()
    rows = []
    for term, row in res.iterrows():
        if term in UNRELIABLE:
            continue  # collinear, uninterpretable -- never the headline term
        fid = int(term[4:].split("_")[0] if term.startswith("act_") else term[len("present_"):])
        rows.append({
            "feature": fid,
            "term": "activation" if term.startswith("act_") else "presence",
            "energy_coef": row["coef"],
            "energy_q": row["q"],
        })
    df = pd.DataFrame(rows)
    # One row per feature: the most significant interpretable term.
    df = df.sort_values("energy_q").drop_duplicates("feature")
    return df, n_designs, r2


def verdict(row):
    e = row["energy_q"] <= Q_STRONG and row["energy_coef"] < 0
    s = row["steer_p"] <= STEER_ALPHA and row["steer_excess_pp"] > 0
    if e and s:
        return "energy + causal steering"
    if e:
        return "energy only"
    if s:
        return "steering only"
    # NB: not the string "null" -- pandas read_csv parses that as NaN.
    return "no detected effect"


def main():
    energy, n_designs, r2 = energy_table()
    steer = steering_table()
    df = energy.merge(steer, on="feature", how="outer")
    df["label"] = df["feature"].map(SHORT_LABELS)
    df["verdict"] = df.apply(verdict, axis=1)
    df = df.sort_values(["energy_q", "steer_p"])

    cols = ["feature", "label", "term", "energy_coef", "energy_q",
            "steer_n_pairs", "steer_feature_rate", "steer_random_rate",
            "steer_excess_pp", "steer_p", "verdict"]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  (energy arm: n={n_designs} designs, R2={r2:.3f})")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
