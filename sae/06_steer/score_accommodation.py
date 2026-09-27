"""Double-difference ProteinMPNN accommodation score for steered mutations.

Asks whether a steered mutation is structurally self-consistent: does the
backbone predicted FOR the mutant sequence prefer the steered residue more
than the native backbone does?

    d_mutant = logP(steered) - logP(native)   on the mutant backbone
    d_native = logP(steered) - logP(native)   on the native backbone
    accommodation = d_mutant - d_native

Positive means the mutant fold accommodates the steered residue better than
the native fold. The paired form cancels each site's intrinsic preferences,
so it works at small n.

Two things this deliberately does NOT do:

  * It does not resample the sequence. ProteinMPNN sampling reads a
    near-native backbone (a single point mutation barely moves the fold) and
    writes back near-native sequence, reverting the mutation before anything
    downstream sees it. That is why the earlier resampling-based check
    returned ~0% for every sparse feature. Use --unconditional_probs_only.
  * It does not condition on the mutant sequence. P(aa | backbone) alone
    keeps the mutation from leaking into its own score.

Expects one .npz per structure from ProteinMPNN's --unconditional_probs_only,
keyed by design_id.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

# ProteinMPNN's fixed output order when the npz carries no alphabet key.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def load_log_probs(path: Path) -> tuple[np.ndarray, str]:
    """Returns (L, A) log-probabilities and the alphabet they are indexed by."""
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    for key in ("log_probs", "logits", "probs"):
        if key in keys:
            arr = np.asarray(data[key], dtype=np.float64)
            break
    else:
        raise SystemExit(f"{path}: no log_probs/logits/probs key (found {sorted(keys)})")

    if arr.ndim == 3:  # (batch, L, A) -- average over decoding orders/samples
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected 2D (L, A) after reduction, got {arr.shape}")

    alphabet = MPNN_ALPHABET
    for key in ("alphabet", "aa_alphabet", "letters"):
        if key in keys:
            raw = data[key]
            alphabet = "".join(raw.tolist()) if raw.dtype.kind in "US" else str(raw)
            break
    if arr.shape[1] != len(alphabet):
        raise SystemExit(f"{path}: {arr.shape[1]} columns but alphabet '{alphabet}' has {len(alphabet)}")

    if key == "probs" or arr.max() > 0:  # probabilities, or logits -- normalize either way
        shifted = arr - arr.max(axis=-1, keepdims=True)
        arr = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return arr, alphabet


def delta(log_probs: np.ndarray, alphabet: str, position: int, steered: str, native: str) -> float:
    if position >= log_probs.shape[0]:
        raise SystemExit(f"position {position} beyond structure length {log_probs.shape[0]}")
    return float(log_probs[position, alphabet.index(steered)] - log_probs[position, alphabet.index(native)])


def sign_test(values: list[float]) -> tuple[int, int, float]:
    """Two-sided exact sign test on how many accommodation scores are positive."""
    from math import comb
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)
    return pos, neg, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True,
                        help="CSV with design_id, position, native_aa and a proposed-residue column")
    parser.add_argument("--proposed-col", default="proposed_aa",
                        help="Column holding the steered residue (use argmax_aa_post for the old lists)")
    parser.add_argument("--native-probs", type=Path, required=True, help="Dir of <design_id>.npz, native backbone")
    parser.add_argument("--mutant-probs", type=Path, required=True, help="Dir of <design_id>.npz, mutant backbone")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.candidates)))
    if args.proposed_col not in rows[0]:
        raise SystemExit(f"--candidates has no '{args.proposed_col}' column (has {list(rows[0])})")

    out, skipped = [], []
    for row in rows:
        design, position = row["design_id"], int(row["position"])
        native, steered = row["native_aa"], row[args.proposed_col]
        if native == steered:
            skipped.append((design, "no_mutation"))
            continue
        nat_path = args.native_probs / f"{design}.npz"
        mut_path = args.mutant_probs / f"{design}.npz"
        if not nat_path.exists() or not mut_path.exists():
            skipped.append((design, "missing_npz"))
            continue

        nat_lp, nat_alpha = load_log_probs(nat_path)
        mut_lp, mut_alpha = load_log_probs(mut_path)
        d_native = delta(nat_lp, nat_alpha, position, steered, native)
        d_mutant = delta(mut_lp, mut_alpha, position, steered, native)
        out.append({
            "design_id": design, "feature_id": row.get("feature_id", ""),
            "resnum": row.get("resnum", ""), "position": position,
            "native_aa": native, "steered_aa": steered,
            "d_native": round(d_native, 4), "d_mutant": round(d_mutant, 4),
            "accommodation": round(d_mutant - d_native, 4),
        })

    if not out:
        raise SystemExit(f"No scoreable sites. Skipped: {skipped}")

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    scores = [r["accommodation"] for r in out]
    pos, neg, p = sign_test(scores)
    print(f"Scored {len(out)} site(s); skipped {len(skipped)}")
    print(f"  mean accommodation  {np.mean(scores):+.3f}  median {np.median(scores):+.3f}")
    print(f"  positive {pos} / negative {neg}   exact sign test p={p:.4f}")
    print(f"\nWrote {args.out}")
    print("Interpretation: positive = the mutant backbone prefers the steered residue more")
    print("than the native backbone does. Needs the random-substitution controls from")
    print("make_control_mutants.py scored the same way before this number means much --")
    print("MPNN recovers native residues ~40-50% of the time, so there is no useful")
    print("absolute baseline without them.")


if __name__ == "__main__":
    main()
