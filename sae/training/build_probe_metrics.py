"""Combine binding-quality metrics (binding_confidence,iptm,ipsae,ipae) across
multiple per-campaign manifest_with_ipsae.csv files into one CSV for
feature_analysis.py's --probe-metrics-csv.

Usage:
    python build_probe_metrics.py --output vilip1_probe_metrics.csv \\
        --source data/vilip1_full20k/manifest_with_ipsae.csv \\
        --source data/vilip1-design-composite_hotspot-combined/manifest_with_ipsae.csv \\
        --source data/vilip1-design-composite_hotspot-20260728/manifest_with_ipsae.csv
"""

import argparse
from pathlib import Path

import pandas as pd

METRIC_COLS = ["binding_confidence", "iptm", "ipsae", "ipae"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], type=Path, required=True, help="Repeatable.")
    args = parser.parse_args()

    frames = []
    for path in args.source:
        df = pd.read_csv(path)
        cols = ["id"] + [c for c in METRIC_COLS if c in df.columns]
        missing = set(METRIC_COLS) - set(cols)
        if missing:
            print(f"{path}: missing columns {missing}, skipping those")
        frames.append(df[cols])

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(combined)} rows")
    present_cols = [c for c in METRIC_COLS if c in combined.columns]
    print(combined[present_cols].notna().sum())


if __name__ == "__main__":
    main()
