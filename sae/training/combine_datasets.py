"""Combine multiple embed_esmc.py --per-residue output directories into one
multi-target training pool for the SAE (Step 3 of the vilip1 roadmap: mix
in UCH-L1/fabp7/REG3A binder designs for training diversity, on top of the
existing vilip1 pool).

Two kinds of input directory:

  --with-sources <dir>
      Already has its own manifest_combined.csv with an (id, source) column
      -- e.g. vilip1_layer23_65k_per_residue, which already distinguishes
      vilip1_full20k / composite_hotspot / composite_hotspot_20260728 /
      natural_binders / binder_dataset_vilip1. Every row's existing source
      label is kept as-is.

  --single-source <dir>:<label>:<sequence_manifest_csv>
      A plain embed_esmc.py --per-residue output (activations.npy +
      index.csv only, no manifest_combined.csv -- e.g. a fresh
      uchl1_layer23_per_residue extraction). Every id in this directory's
      index.csv gets the given constant source label. <sequence_manifest_csv>
      is the ORIGINAL campaign manifest (e.g. data/UCH_L1_final/manifest.csv)
      with id,sequence columns -- needed because embed_esmc.py's per-residue
      output never stores sequence text itself, and benchmark.py (unlike
      train.py) needs real sequences to re-run the actual ESM-C forward pass
      for both models, not just id/source.

All directories must be embeddings from the SAME ESM-C layer (layer 23,
matching vilip1's existing extraction and Biohub's own general-purpose SAE
-- see sae/README.md) for the combined pool to be meaningful; this script
does not verify that itself, since layer identity isn't recorded in the
per-residue output format.

Output directory gets the same activations.npy/index.csv/manifest_combined.csv
layout as any other per-residue dir, so it's a drop-in --data-dir for both
train.py (only reads id/source) and benchmark.py (also reads sequence).

Streams activations.npy chunk-by-chunk between memmaps (never loads a full
input or the full output into RAM), since combining ~5 datasets each
~10-15GB adds up fast.

Usage:
    python combine_datasets.py --output combined_multi_target_layer23_per_residue \\
        --with-sources vilip1_layer23_65k_per_residue \\
        --single-source uchl1_layer23_per_residue:uchl1:data/UCH_L1_final/manifest.csv \\
        --single-source fabp7_layer23_per_residue:fabp7:data/fabp7_full20k/manifest.csv \\
        --single-source reg3a_45_57_layer23_per_residue:reg3a_45_57:data/REG3A_45_57/manifest.csv \\
        --single-source reg3a_25_33_layer23_per_residue:reg3a_25_33:data/REG3A_25_33/manifest.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COPY_CHUNK_ROWS = 200_000  # residues per chunk when streaming activations.npy


def load_source_labeled_index(
    data_dir: Path, source_label: str | None, sequence_manifest: Path | None
) -> pd.DataFrame:
    """Returns a DataFrame with id, start, length, source, sequence columns
    for one input directory -- source_label is None for --with-sources dirs
    (source pulled from that dir's own manifest_combined.csv, which also
    already has sequence). sequence_manifest is required for --single-source
    dirs (their per-residue output has no sequence text of its own)."""
    index_df = pd.read_csv(data_dir / "index.csv")
    if source_label is None:
        manifest_df = pd.read_csv(data_dir / "manifest_combined.csv")[["id", "source", "sequence"]]
        merged = index_df.merge(manifest_df, on="id", how="left")
        assert merged["source"].notna().all(), (
            f"{data_dir}: some index.csv ids missing from manifest_combined.csv"
        )
    else:
        assert sequence_manifest is not None, f"{data_dir}: --single-source needs a sequence manifest"
        seq_df = pd.read_csv(sequence_manifest)[["id", "sequence"]]
        merged = index_df.merge(seq_df, on="id", how="left")
        assert merged["sequence"].notna().all(), (
            f"{data_dir}: some index.csv ids missing from {sequence_manifest}"
        )
        merged["source"] = source_label
    return merged[["id", "start", "length", "source", "sequence"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--with-sources", action="append", default=[], type=Path,
        help="Per-residue dir that already has its own manifest_combined.csv (source column kept as-is). Repeatable.",
    )
    parser.add_argument(
        "--single-source", action="append", default=[],
        help="<dir>:<source_label>:<sequence_manifest_csv> -- plain per-residue dir, every id "
        "labeled with the given constant source, sequences pulled from the given original "
        "campaign manifest (id,sequence columns). Repeatable.",
    )
    args = parser.parse_args()

    inputs: list[tuple[Path, str | None, Path | None]] = [(d, None, None) for d in args.with_sources]
    for spec in args.single_source:
        parts = spec.split(":")
        assert len(parts) == 3, f"--single-source expects <dir>:<label>:<sequence_manifest_csv>, got {spec!r}"
        dir_str, label, seq_manifest_str = parts
        assert label, f"--single-source expects <dir>:<label>:<sequence_manifest_csv>, got {spec!r}"
        inputs.append((Path(dir_str), label, Path(seq_manifest_str)))
    assert inputs, "need at least one --with-sources or --single-source input"

    per_input_index = []
    total_rows = 0
    d_model = None
    for data_dir, label, seq_manifest in inputs:
        idx = load_source_labeled_index(data_dir, label, seq_manifest)
        acts = np.load(data_dir / "activations.npy", mmap_mode="r")
        assert acts.shape[0] >= idx["start"].add(idx["length"]).max(), (
            f"{data_dir}: index.csv references rows beyond activations.npy's length"
        )
        if d_model is None:
            d_model = acts.shape[1]
        else:
            assert acts.shape[1] == d_model, (
                f"{data_dir}: hidden_dim={acts.shape[1]} doesn't match earlier input's {d_model} -- "
                "check every input is from the same ESM-C layer/model"
            )
        n_residues = int(idx["length"].sum())
        print(f"{data_dir} (source={label or 'from its own manifest_combined.csv'}): "
              f"{len(idx)} proteins, {n_residues} residues")
        per_input_index.append((data_dir, idx, n_residues))
        total_rows += n_residues

    print(f"Total: {total_rows} residues across {sum(len(i) for _, i, _ in per_input_index)} proteins, d_model={d_model}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_acts = np.lib.format.open_memmap(
        args.output / "activations.npy", mode="w+", dtype=np.float16, shape=(total_rows, d_model)
    )

    combined_index_rows = []
    combined_manifest_rows = []
    cursor = 0
    for data_dir, idx, n_residues in per_input_index:
        in_acts = np.load(data_dir / "activations.npy", mmap_mode="r")
        for chunk_start in range(0, n_residues, COPY_CHUNK_ROWS):
            chunk_end = min(chunk_start + COPY_CHUNK_ROWS, n_residues)
            out_acts[cursor + chunk_start : cursor + chunk_end] = in_acts[chunk_start:chunk_end]

        for _, row in idx.iterrows():
            new_start = cursor + row["start"]
            combined_index_rows.append({"id": row["id"], "start": new_start, "length": row["length"]})
            combined_manifest_rows.append({"id": row["id"], "sequence": row["sequence"], "source": row["source"]})

        cursor += n_residues
        print(f"  copied {data_dir} -> offset {cursor - n_residues}..{cursor}")

    out_acts.flush()
    assert cursor == total_rows

    pd.DataFrame(combined_index_rows).to_csv(args.output / "index.csv", index=False)
    manifest_out = pd.DataFrame(combined_manifest_rows)
    manifest_out.to_csv(args.output / "manifest_combined.csv", index=False)
    print(f"Wrote {args.output}/activations.npy, index.csv, manifest_combined.csv")
    print("Source counts:\n", manifest_out["source"].value_counts())


if __name__ == "__main__":
    main()
