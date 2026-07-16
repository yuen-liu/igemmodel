"""Rank a downloaded Boltz protein-design campaign and write a top-N manifest.

Two input modes:
  --results-dir   walks `<results_dir>/pres_*/metadata.json` (the per-result
                  files boltz-api's download-results writes to disk). Only
                  covers results whose metadata.json actually downloaded.
  --index-jsonl   reads `results/index.jsonl` directly (one generated design
                  per line, per the boltz-protein-design skill's documented
                  layout) -- covers every generated design regardless of
                  which per-result files happened to download, so prefer
                  this when the file exists.

Ranks by binding_confidence (tiebreakers: iptm desc, min_interaction_pae asc
-- same convention as the boltz_*_experiments.ipynb notebooks), and writes a
CSV manifest with id/sequence/metrics for the top N.

Usage:
    python rank_boltz_results.py --results-dir /path/to/campaign/results \
        --top-n 500 --output manifest.csv
    python rank_boltz_results.py --index-jsonl /path/to/campaign/results/index.jsonl \
        --top-n 500 --output manifest.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _record_from_item(item: dict) -> dict | None:
    binder_seq = next(
        (e["value"] for e in item.get("entities", []) if "C" in e.get("chain_ids", [])),
        None,
    )
    if binder_seq is None:
        return None
    m = item.get("metrics", {})
    return {
        "id": item["id"],
        "sequence": binder_seq,
        "binding_confidence": m.get("binding_confidence"),
        "iptm": m.get("iptm"),
        "min_interaction_pae": m.get("min_interaction_pae"),
        "structure_confidence": m.get("structure_confidence"),
        "complex_plddt": m.get("complex_plddt"),
    }


def load_results_from_metadata(results_dir: Path) -> pd.DataFrame:
    records = []
    for meta_path in sorted(results_dir.glob("pres_*/metadata.json")):
        item = json.loads(meta_path.read_text())
        record = _record_from_item(item)
        if record is None:
            print(f"Skipping {meta_path}: no chain C entity found")
            continue
        records.append(record)
    if not records:
        raise ValueError(f"No pres_*/metadata.json files found under {results_dir}")
    return pd.DataFrame(records)


def load_results_from_index(index_path: Path) -> pd.DataFrame:
    records = []
    with open(index_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            record = _record_from_item(item)
            if record is None:
                print(f"Skipping line {line_num}: no chain C entity found")
                continue
            records.append(record)
    if not records:
        raise ValueError(f"No usable records found in {index_path}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--index-jsonl", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("manifest.csv"))
    args = parser.parse_args()

    if args.index_jsonl is not None:
        df = load_results_from_index(args.index_jsonl)
        print(f"Loaded {len(df)} results from {args.index_jsonl}")
    elif args.results_dir is not None:
        df = load_results_from_metadata(args.results_dir)
        print(f"Loaded {len(df)} results from {args.results_dir}")
    else:
        parser.error("must pass either --results-dir or --index-jsonl")

    n_missing = df["binding_confidence"].isna().sum()
    if n_missing:
        print(f"Warning: {n_missing} results missing binding_confidence, dropping them")
        df = df.dropna(subset=["binding_confidence"])

    df = df.sort_values(
        ["binding_confidence", "iptm", "min_interaction_pae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top = df.head(args.top_n)
    if len(top) < args.top_n:
        print(
            f"Warning: only {len(top)} results available, "
            f"fewer than requested top-{args.top_n}"
        )

    top.to_csv(args.output, index=False)
    print(f"Wrote {len(top)} ranked designs to {args.output}")
    print(top[["id", "binding_confidence", "iptm", "min_interaction_pae"]].head(10))


if __name__ == "__main__":
    main()
