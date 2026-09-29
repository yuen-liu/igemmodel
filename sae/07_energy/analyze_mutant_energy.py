"""Paired test: does installing feature 233's anchor change interaction energy?

Consumes the per-structure CSVs written by run_mutant_energy_profiling.bash
and reports one effect per design, so the 20 designs are 20 independent
observations rather than 100 correlated rows.

**The primary comparison is mutant vs. its own random-substitution controls,
not mutant vs. native.** The mutant backbone was folded by Boltz *from the
mutant sequence*, so the substitution is baked into the geometry before Prime
ever reads it -- exactly the leak that made `d_mutant` unquotable in the
ProteinMPNN accommodation arm (RESULTS.md, "Structural arm results"). The
controls are folded the same way from their own substituted sequences, so the
artifact cancels in mutant-minus-control. Mutant-minus-native does not cancel
it and is reported only as a confounded secondary.

Usage:
    python analyze_mutant_energy.py
    python analyze_mutant_energy.py --results-dir mutant_energy_results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=HERE / "mutant_energy_results")
    ap.add_argument("--manifest", type=Path, default=HERE / "mutant_energy_manifest.csv")
    ap.add_argument("--out", type=Path, default=HERE / "mutant_energy_paired.csv")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"{args.manifest} not found -- run build_mutant_energy_inputs.py first.")
    manifest = pd.read_csv(args.manifest)

    csvs = sorted(args.results_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No per-structure CSVs in {args.results_dir}.")
    energy = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # One row per structure: profile_binding_energy.py repeats the same energy
    # per feature listed, and these entries list exactly one.
    energy = energy.drop_duplicates("design_id")[
        ["design_id", "dE_interaction", "n_interface_hbonds"]
    ].rename(columns={"design_id": "structure_id"})

    df = manifest.merge(energy, on="structure_id", how="left")
    missing = df["dE_interaction"].isna().sum()
    print(f"{len(df)} structures in manifest, {len(df) - missing} with an energy result")
    if missing:
        print(f"  {missing} structure(s) have no CSV -- excluded")
        df = df.dropna(subset=["dE_interaction"])

    rows = []
    for design, g in df.groupby("design_id"):
        arms = {a: sub for a, sub in g.groupby("arm")}
        if "mutant" not in arms or "control" not in arms:
            print(f"  skipping {design}: needs both a mutant and >=1 control")
            continue
        mut = arms["mutant"]["dE_interaction"].mean()
        ctl = arms["control"]["dE_interaction"].mean()
        nat = arms["native"]["dE_interaction"].mean() if "native" in arms else np.nan
        rows.append({
            "design_id": design,
            "dE_mutant": mut,
            "dE_control_mean": ctl,
            "n_controls": len(arms["control"]),
            "dE_native": nat,
            "ddE_vs_control": mut - ctl,
            "ddE_vs_native": mut - nat,
        })

    paired = pd.DataFrame(rows)
    if paired.empty:
        raise SystemExit("No design produced a complete paired set.")
    paired.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(paired)} paired designs)\n")

    def report(col, label, primary):
        x = paired[col].dropna()
        if len(x) < 3:
            print(f"{label}: n={len(x)}, too few to test")
            return
        t, p_t = stats.ttest_1samp(x, 0.0)
        try:
            _, p_w = stats.wilcoxon(x)
        except ValueError:
            p_w = float("nan")
        d = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else float("nan")
        tag = "PRIMARY" if primary else "secondary, CONFOUNDED by the Boltz refold"
        print(f"{label}  [{tag}]")
        print(f"  n={len(x)} designs   mean={x.mean():+.2f} kcal/mol   median={x.median():+.2f}")
        print(f"  t-test p={p_t:.4f}   Wilcoxon p={p_w:.4f}   Cohen's d={d:+.2f}")
        print(f"  negative = steered mutation binds more favorably than the comparison\n")

    report("ddE_vs_control", "mutant - random-substitution controls", True)
    report("ddE_vs_native", "mutant - native", False)

    # Diagnostic. The native was never steered, so native-minus-control should
    # sit at zero. If it does not, the control set carries a systematic offset
    # -- in the 2026-09-29 run, controls installed bulky residues (W, I, Q, F)
    # while every mutant installed K or E, and Prime's interaction energy
    # tracks how much residue sits at the interface. Any mutant-minus-control
    # effect smaller than this offset is measuring the amino acid, not the
    # feature.
    nat = (paired["dE_native"] - paired["dE_control_mean"]).dropna()
    if len(nat) >= 3:
        _, p_nat = stats.ttest_1samp(nat, 0.0)
        print(f"DIAGNOSTIC  native - controls: mean={nat.mean():+.2f} kcal/mol, p={p_nat:.4f}")
        print("  Should be ~0. A non-zero offset here is a property of the control")
        print("  set, not of steering -- compare it against the primary effect above.\n")

    # Power. The between-design energy association this arm was built to probe
    # is -13.5 kcal/mol per SD of max_activation.
    sd = paired["ddE_vs_control"].std(ddof=1)
    n = paired["ddE_vs_control"].notna().sum()
    if n >= 3 and sd > 0:
        mde = 2.87 * sd / np.sqrt(n)
        print(f"POWER  paired SD={sd:.1f} kcal/mol, n={n} -> minimum detectable effect "
              f"~{mde:.0f} kcal/mol (80% power, two-sided 0.05)\n")

    print("Interpreting a null here: the regression predictor is max_activation,")
    print("the MAXIMUM over positions in a design. These designs already carry 7-21")
    print("233 sites each, so installing one more anchor moves max_activation by")
    print("roughly nothing -- the model's own predicted effect for this mutation is")
    print("~0, not -13.5. A null is consistent with the energy association being real.")
    print("This tests a within-design single-residue perturbation, not the")
    print("between-design association.\n")

    print("Reminder: dE_interaction is a Prime chain-split interaction energy, not a")
    print("binding free energy and not a folding/thermostability measure.")


if __name__ == "__main__":
    main()
