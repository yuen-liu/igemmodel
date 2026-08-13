"""One-off fix-up: add a `sequence` column to a manifest_combined.csv that's
missing one (e.g. combine_datasets.py output from before it carried sequence
through -- see that script's docstring). benchmark.py needs `sequence`
(train.py doesn't), so a combined manifest built with the old version of
combine_datasets.py will crash benchmark.py with `KeyError: 'sequence'`.

Overwrites --combined-manifest in place after joining in `sequence` from one
or more --sequence-source CSVs (each needs id,sequence columns at least --
extra columns are ignored). If an id appears in more than one source, the
first source listed wins.

Usage:
    python patch_manifest_sequence.py \\
        --combined-manifest ~/embeddings/vilip1_all_layer23_per_residue/manifest_combined.csv \\
        --sequence-source ~/notebooks/vilip1_layer23_65k_per_residue/manifest_combined.csv \\
        --sequence-source ~/notebooks/sae_training/data/UCH_L1_final/manifest.csv \\
        --sequence-source ~/notebooks/sae_training/data/fabp7_full20k/manifest.csv \\
        --sequence-source ~/notebooks/sae_training/data/REG3A_45_57/manifest.csv \\
        --sequence-source ~/notebooks/sae_training/data/REG3A_25_33/manifest.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument(
        "--sequence-source", action="append", default=[], type=Path, required=True,
        help="CSV with at least id,sequence columns. Repeatable -- first source listed wins on duplicate ids.",
    )
    args = parser.parse_args()

    combined = pd.read_csv(args.combined_manifest)
    if "sequence" in combined.columns:
        print(f"{args.combined_manifest} already has a sequence column -- nothing to do.")
        return

    seq_frames = [pd.read_csv(p)[["id", "sequence"]] for p in args.sequence_source]
    seq_df = pd.concat(seq_frames, ignore_index=True).drop_duplicates("id", keep="first")

    patched = combined.merge(seq_df, on="id", how="left")
    n_missing = int(patched["sequence"].isna().sum())
    assert n_missing == 0, (
        f"{n_missing} ids in {args.combined_manifest} have no sequence in any --sequence-source -- "
        "check every source dataset was listed"
    )

    other_cols = [c for c in combined.columns if c != "id"]
    patched = patched[["id", "sequence"] + other_cols]
    patched.to_csv(args.combined_manifest, index=False)
    print(f"Patched {args.combined_manifest}: {len(patched)} rows now have sequence")


if __name__ == "__main__":
    main()
