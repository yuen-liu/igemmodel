"""Build a manifest_combined.csv (id,sequence,source) for a per-residue
directory that only has activations.npy + index.csv -- e.g.
embed_esmc_paired.py's output, which (like embed_esmc.py) never writes a
manifest_combined.csv itself. Needed before train.py or benchmark.py can
use the directory as --data-dir.

Joins index.csv's id list against one or more --reference-manifest CSVs
(each with id,sequence,source -- e.g. the original
vilip1_layer23_65k_per_residue/manifest_combined.csv) to pull in sequence
and source for whichever subset of ids is actually present in this
directory's index.csv.

Usage:
    python build_manifest_from_index.py \\
        --data-dir ~/embeddings/vilip1_binder_plus_target_layer23_per_residue \\
        --reference-manifest ~/notebooks/vilip1_layer23_65k_per_residue/manifest_combined.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest", action="append", default=[], type=Path, required=True,
        help="CSV with id,sequence,source columns. Repeatable -- first source listed wins on duplicate ids.",
    )
    args = parser.parse_args()

    index_df = pd.read_csv(args.data_dir / "index.csv")
    ref_frames = [pd.read_csv(p)[["id", "sequence", "source"]] for p in args.reference_manifest]
    ref_df = pd.concat(ref_frames, ignore_index=True).drop_duplicates("id", keep="first")

    manifest = index_df[["id"]].merge(ref_df, on="id", how="left")
    n_missing = int(manifest["sequence"].isna().sum())
    assert n_missing == 0, (
        f"{n_missing}/{len(manifest)} ids in {args.data_dir}/index.csv have no match in any "
        "--reference-manifest -- check the right reference file(s) were given"
    )

    out_path = args.data_dir / "manifest_combined.csv"
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(manifest)} rows")
    print("Source counts:\n", manifest["source"].value_counts())


if __name__ == "__main__":
    main()
