"""Pick the exact set of designs needed for energy profiling, before
downloading/transferring any structures.

Andrew's Drive folders contain the full ~65k-design campaign; we only need
~50 designs per candidate feature (per sae/RESULTS.md's steering-section
gap note). Downloading everything first is wasteful -- this script reuses
the same sampling logic already written and tested for
sae/06_steer/inject_feature.py (select_design_feature_pairs) against the
already-merged sae/results/run4/yamls/interface_features_combined.yaml,
and writes out just the design ids actually needed plus their expected
filename (`{design_id}_predicted.cif`, matching the flat naming Andrew's
folders actually use -- confirmed against pres_0A0NZAOnP7bs1ZD5CRnY_predicted.cif
on Gates).

Usage:
    python select_energy_sample.py \\
        --interface-yaml ../results/run4/yamls/interface_features_combined.yaml \\
        --designs-per-feature 50 --seed 0 \\
        --output energy_sample_design_ids.txt
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_steer"))
from inject_feature import select_design_feature_pairs, YAML_LOADER  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface-yaml", type=Path, required=True)
    parser.add_argument("--designs-per-feature", type=int, default=50)
    parser.add_argument("--sampling", choices=["top", "random"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Plain-text file, one design id per line")
    args = parser.parse_args()

    print(f"Loading {args.interface_yaml}...")
    with open(args.interface_yaml) as f:
        yaml_data = yaml.load(f, Loader=YAML_LOADER)

    feature_ids = yaml_data["metadata"]["features_checked"]
    selected = select_design_feature_pairs(yaml_data, feature_ids, args.designs_per_feature, args.sampling, args.seed)

    all_design_ids = set()
    for feature_id, entries in selected.items():
        for entry in entries:
            all_design_ids.add(entry["design_id"])

    with open(args.output, "w") as f:
        for design_id in sorted(all_design_ids):
            f.write(design_id + "\n")

    print(f"\n{len(all_design_ids)} unique design(s) needed across {len(feature_ids)} feature(s) "
          f"(some designs may satisfy more than one feature's sample).")
    print(f"Expected filenames: {{design_id}}_predicted.cif (matches Andrew's flat naming convention).")
    print(f"Wrote design ids to {args.output}")


if __name__ == "__main__":
    main()
