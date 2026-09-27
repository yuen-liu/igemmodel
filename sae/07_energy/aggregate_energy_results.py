"""Concatenate profile_binding_energy.py's per-design CSVs into one file,
and print a per-feature summary (mean/median dE_interaction, n_designs) --
the actual "features with high delta binding energy" ranking the original
steering plan calls for.

Usage:
    python aggregate_energy_results.py --results-dir energy_results \\
        --design-ids-file energy_sample_design_ids.txt \\
        --output energy_profile_combined.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--design-ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_ids = [line.strip() for line in open(args.design_ids_file) if line.strip()]
    dfs = []
    missing = []
    for design_id in expected_ids:
        csv_path = args.results_dir / f"{design_id}.csv"
        if not csv_path.exists():
            missing.append(design_id)
            continue
        dfs.append(pd.read_csv(csv_path))

    if missing:
        print(f"WARNING: {len(missing)}/{len(expected_ids)} design(s) missing a result file "
              f"(SLURM task failed or still running?), e.g.: {missing[:5]}")

    if not dfs:
        raise SystemExit("No result files found -- nothing to aggregate.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} row(s) ({combined['design_id'].nunique()} unique design(s)) to {args.output}")

    print("\nPer-feature summary (sorted most-favorable dE_interaction first):")
    summary = combined.groupby("feature_id").agg(
        n_designs=("design_id", "nunique"),
        mean_dE_interaction=("dE_interaction", "mean"),
        median_dE_interaction=("dE_interaction", "median"),
        mean_n_interface_hbonds=("n_interface_hbonds", "mean"),
    ).reset_index().sort_values("mean_dE_interaction")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
