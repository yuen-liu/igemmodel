"""Pick a foldable, statistically clean subset for the structural arm.

The anchor-aware list holds 264 sites for feature 233 but spread over only 20
designs, 7-21 sites each. Sites within one design share a backbone, so they
are not independent observations: a sign test over them would treat correlated
measurements as independent and overstate significance. This takes ONE site
per design, giving independent paired observations, and keeps the fold budget
small enough to actually run (n designs mutant folds + n natives + n*k
controls).

Preference order within a design: lowest top-k rank (the readout's strongest
anchor-installing call), then lowest alpha (smallest intervention that
produced it), then lowest resnum for determinism. Terminal positions are
dropped -- ESM-C's predictions there are least reliable and an initiator
methionine is not a mutable site.
"""

import argparse
import csv
import random
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--min-resnum", type=int, default=3,
                        help="Drop positions below this; termini are unreliable and M1 is not mutable")
    parser.add_argument("--controls-per-site", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-sites", type=Path, required=True)
    parser.add_argument("--out-fasta", type=Path, required=True)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.candidates))
            if r["feature_id"] == args.feature and int(r["resnum"]) >= args.min_resnum]
    if not rows:
        raise SystemExit(f"No rows for feature {args.feature} at resnum >= {args.min_resnum}")

    best: dict[str, tuple] = {}
    for row in rows:
        rank = int(row["selection_reason"].rsplit("rank", 1)[1])
        key = (rank, float(row["alpha_multiplier"]), int(row["resnum"]))
        if row["design_id"] not in best or key < best[row["design_id"]][0]:
            best[row["design_id"]] = (key, row)

    sites = [row for _, row in sorted(best.values(), key=lambda kv: kv[1]["design_id"])]

    rng = random.Random(args.seed)
    fasta, controls = [], []
    for row in sites:
        position = int(row["position"])
        mutant = row["mutated_sequence"]
        native = mutant[:position] + row["native_aa"] + mutant[position + 1:]
        assert mutant[position] == row["proposed_aa"]
        fasta.append((f"{row['design_id']}__native", native))
        fasta.append((f"{row['design_id']}__mutant_{row['native_aa']}{row['resnum']}{row['proposed_aa']}", mutant))
        pool = [a for a in AA if a not in (row["native_aa"], row["proposed_aa"])]
        for control_aa in rng.sample(pool, args.controls_per_site):
            seq = native[:position] + control_aa + native[position + 1:]
            fasta.append((f"{row['design_id']}__control_{row['native_aa']}{row['resnum']}{control_aa}", seq))
            controls.append({**row, "control_aa": control_aa, "control_sequence": seq})

    cols = ["design_id", "feature_id", "resnum", "position", "native_aa",
            "argmax_aa_post", "proposed_aa", "selection_reason", "top_k_aa_post",
            "alpha_multiplier", "mutated_sequence"]
    with args.out_sites.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sites)

    with args.out_fasta.open("w") as handle:
        for name, seq in fasta:
            handle.write(f">{name}\n{seq}\n")

    print(f"Feature {args.feature}: {len(rows)} eligible proposals -> {len(sites)} sites (one per design)")
    print(f"  fold budget: {len(sites)} native + {len(sites)} mutant + "
          f"{len(sites) * args.controls_per_site} control = {len(fasta)} structures")
    print(f"  ranks used: {sorted({r['selection_reason'] for r in sites})}")
    print(f"\nWrote {args.out_sites} and {args.out_fasta}")


if __name__ == "__main__":
    main()
