"""Generate random-substitution control sequences for the structural test.

The double-difference MPNN score (accommodation of the steered residue on the
mutant backbone vs the native backbone) is only interpretable against a null:
does *any* substitution at the same position score the same way? This writes,
per candidate site, N random substitutions drawn from the 18 amino acids that
are neither the native nor the steered residue, so each control is matched to
its site and differs only in which residue was installed.
"""

import argparse
import csv
import random
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--n-per-site", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-fasta", type=Path, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = list(csv.DictReader(open(args.candidates)))
    out = []

    for row in rows:
        position = int(row["position"])
        native_aa, steered_aa = row["native_aa"], row["argmax_aa_post"]
        mutant = row["mutated_sequence"]
        # Reconstruct native from the mutant: they differ only at `position`.
        native_seq = mutant[:position] + native_aa + mutant[position + 1:]
        assert mutant[position] == steered_aa

        choices = [a for a in AA if a not in (native_aa, steered_aa)]
        for control_aa in rng.sample(choices, args.n_per_site):
            out.append({
                "control_id": f"{row['design_id']}_{row['resnum']}{control_aa}",
                "design_id": row["design_id"],
                "feature_id": row["feature_id"],
                "resnum": row["resnum"],
                "position": position,
                "native_aa": native_aa,
                "steered_aa": steered_aa,
                "control_aa": control_aa,
                "sequence": native_seq[:position] + control_aa + native_seq[position + 1:],
            })

    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    # Every control must differ from native and from the steered mutant at the
    # target position only -- otherwise it is not a null for this site.
    for r in out:
        p = r["position"]
        assert r["sequence"][p] == r["control_aa"]
        assert r["control_aa"] not in (r["native_aa"], r["steered_aa"])

    if args.out_fasta:
        with args.out_fasta.open("w") as handle:
            for r in out:
                handle.write(f">{r['control_id']}\n{r['sequence']}\n")

    print(f"Wrote {len(out)} control sequences ({len(rows)} sites x {args.n_per_site}) to {args.out_csv}")
    if args.out_fasta:
        print(f"FASTA for folding: {args.out_fasta}")


if __name__ == "__main__":
    main()
