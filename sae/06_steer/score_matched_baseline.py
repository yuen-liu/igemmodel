"""Composition-matched test of whether a feature's positions favour its anchors.

The unmatched version of this comparison is confounded. The score

    s(pos) = log P(anchor residues | backbone) - log P(native residue at pos)

is measured *against whatever the native residue is*, so it is near zero
wherever the native residue is already an anchor. These designs are ~45% E+K
by composition, and the anchor-aware selector deliberately keeps only sites
whose native residue is NOT an anchor. So feature positions (never E/K) get
compared against a baseline that is ~half E/K-and-therefore-pinned-near-zero,
and a positive result follows from the selection alone.

This holds the native residue fixed: each feature site is compared only
against non-feature positions carrying the SAME native residue, in the same
structure where possible and pooled across structures when a residue is thin.

Reports both comparisons. The unmatched number is the confounded one; the gap
between them is the size of the composition artifact.

Needs only the NATIVE backbone probabilities -- no mutant structures, so no
Boltz-conditioning leak: a native backbone was folded from the native sequence
and nothing downstream of any mutation touches it.
"""

import argparse
import csv
from collections import defaultdict
from math import comb, log, exp
from pathlib import Path

import numpy as np

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def locate_binder(data, native_seq: str, alphabet: str) -> int:
    """Index at which `native_seq` starts inside the structure's sequence.

    These are binder-vs-target complexes, so a structure holds the Vilip-1
    target (191 residues) followed by the binder: log_p is (1, 256, 21) for a
    65-residue binder. Site positions are binder-local, and -- more easily
    missed -- the non-site BASELINE must be restricted to the binder too, or it
    silently averages over target-chain positions instead.
    """
    if "S" not in data.files:
        return 0
    seq = "".join(alphabet[i] for i in np.asarray(data["S"]).ravel().tolist())
    offset = seq.find(native_seq)
    if offset < 0:
        raise SystemExit(
            "Could not locate the binder sequence inside the structure's own sequence; "
            "refusing to guess the chain offset."
        )
    return offset


def load_log_probs(path: Path) -> tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    for key in ("log_probs", "log_p", "logits", "probs"):
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float64)
            break
    else:
        raise SystemExit(f"{path}: no log_probs/logits/probs (found {sorted(data.files)})")
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    alphabet = MPNN_ALPHABET
    for key in ("alphabet", "aa_alphabet", "letters"):
        if key in data.files:
            raw = data[key]
            alphabet = "".join(raw.tolist()) if raw.dtype.kind in "US" else str(raw)
            break
    if arr.shape[1] != len(alphabet):
        raise SystemExit(f"{path}: {arr.shape[1]} cols vs alphabet '{alphabet}'")
    shifted = arr - arr.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)), alphabet


AA20 = "ACDEFGHIKLMNPQRSTVWY"


def score(lp: np.ndarray, alphabet: str, pos: int, anchors: str, native_aa: str,
          reference: str = "native", n_ref: int = 3, rng=None) -> float:
    """log P(any anchor) minus a reference, at one position.

    reference="native" -> minus log P(native residue). Asks "is the anchor
        preferred over what is actually there".
    reference="random" -> minus the mean log P of n_ref random residues that
        are neither the native nor an anchor. Asks "is the anchor preferred
        over an arbitrary alternative" -- which is the quantity the
        steered-vs-random-control comparison computes, since log P(native)
        cancels out of it.

    The random reference needs the SAME matched treatment as the native one.
    In a corpus that is ~45% E+K, log P(E/K) exceeds the mean log P of three
    arbitrary residues at most surface positions regardless of any feature, so
    a positive value alone says nothing; only its excess over matched
    non-feature positions does.
    """
    p_anchor = sum(exp(lp[pos, alphabet.index(a)]) for a in anchors if a in alphabet)
    if p_anchor <= 0:
        return float("-inf")
    if reference == "native":
        return log(p_anchor) - float(lp[pos, alphabet.index(native_aa)])
    pool = [a for a in AA20 if a not in anchors and a != native_aa and a in alphabet]
    picks = rng.sample(pool, min(n_ref, len(pool)))
    return log(p_anchor) - float(np.mean([lp[pos, alphabet.index(a)] for a in picks]))


def sign_test(values) -> tuple[int, int, float]:
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    return pos, neg, min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=Path, required=True,
                    help="structural_sites_*.csv: design_id, position, native_aa, mutated_sequence")
    ap.add_argument("--native-probs", type=Path, required=True, help="Dir of <design_id>.npz, NATIVE backbones")
    ap.add_argument("--anchors", default="EK", help="Anchor residues for this feature (233 -> EK)")
    ap.add_argument("--reference", choices=["native", "random"], default="native",
                    help="native: log P(anchor) - log P(native residue). random: minus the mean "
                         "log P of --n-ref arbitrary non-native non-anchor residues, which is what "
                         "the steered-vs-control comparison measures. Either way the matched "
                         "contrast is the test; the raw value is confounded by composition.")
    ap.add_argument("--n-ref", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import random as _random
    sites = list(csv.DictReader(open(args.sites)))
    anchors = args.anchors.strip().upper()
    rng = _random.Random(args.seed)
    sc = lambda lp, al, p, aa: score(lp, al, p, anchors, aa, args.reference, args.n_ref, rng)

    # Baseline pool: every non-site position, keyed by its native residue.
    pool_by_aa: dict[str, list[float]] = defaultdict(list)
    pool_by_design_aa: dict[tuple[str, str], list[float]] = defaultdict(list)
    site_scores, missing, offsets = [], [], {}
    site_positions = defaultdict(set)
    for r in sites:
        site_positions[r["design_id"]].add(int(r["position"]))

    for r in sites:
        design, pos = r["design_id"], int(r["position"])
        path = args.native_probs / f"{design}.npz"
        if not path.exists():
            missing.append(design)
            continue
        lp, alphabet = load_log_probs(path)
        mutant = r["mutated_sequence"]
        native_seq = mutant[:pos] + r["native_aa"] + mutant[pos + 1:]
        offset = locate_binder(np.load(path, allow_pickle=True), native_seq, alphabet)
        if offset + len(native_seq) > lp.shape[0]:
            raise SystemExit(f"{design}: binder at offset {offset} overruns structure {lp.shape[0]}")
        assert native_seq[pos] == r["native_aa"]
        offsets[design] = offset
        site_scores.append({
            "design_id": design, "resnum": r.get("resnum", ""), "position": pos,
            "native_aa": r["native_aa"], "chain_offset": offset,
            "site_score": sc(lp, alphabet, offset + pos, r["native_aa"]),
        })
        # Baseline over the BINDER only -- never the target chain.
        for i, aa in enumerate(native_seq):
            if i in site_positions[design] or aa not in alphabet:
                continue
            s = sc(lp, alphabet, offset + i, aa)
            pool_by_aa[aa].append(s)
            pool_by_design_aa[(design, aa)].append(s)

    if missing:
        print(f"WARNING: missing npz for {len(set(missing))} design(s): {sorted(set(missing))[:5]}")
    if not site_scores:
        raise SystemExit("No sites scored.")

    rows = []
    for s in site_scores:
        aa = s["native_aa"]
        local = pool_by_design_aa.get((s["design_id"], aa), [])
        pooled = pool_by_aa.get(aa, [])
        matched = local if len(local) >= 3 else pooled
        source = "same_design" if len(local) >= 3 else ("pooled_across_designs" if pooled else "none")
        all_other = [v for k, vs in pool_by_design_aa.items() if k[0] == s["design_id"] for v in vs]
        rows.append({
            **s,
            "matched_baseline": round(float(np.mean(matched)), 4) if matched else "",
            "matched_n": len(matched), "matched_source": source,
            "matched_delta": round(s["site_score"] - float(np.mean(matched)), 4) if matched else "",
            "unmatched_baseline": round(float(np.mean(all_other)), 4) if all_other else "",
            "unmatched_delta": round(s["site_score"] - float(np.mean(all_other)), 4) if all_other else "",
        })

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def report(label, key):
        vals = [r[key] for r in rows if r[key] != ""]
        if not vals:
            print(f"  {label}: no data")
            return
        p, n, pv = sign_test(vals)
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        d = float(np.mean(vals)) / sd if sd and sd == sd and sd > 0 else float("nan")
        print(f"  {label:<34} mean {np.mean(vals):+.3f}  d={d:+.2f}  "
              f"{p}+/{n}-  sign test p={pv:.4f}")

    print(f"\nScored {len(rows)} site(s), anchors '{anchors}', reference '{args.reference}'")
    if offsets:
        print(f"binder chain offsets detected: {sorted(set(offsets.values()))} "
              f"(baseline restricted to the binder, target chain excluded)")
    print(f"baseline pool sizes by native residue: "
          f"{ {aa: len(v) for aa, v in sorted(pool_by_aa.items())} }")
    print("\nIs the anchor favoured at feature sites, relative to...")
    report("ALL other positions (CONFOUNDED)", "unmatched_delta")
    report("same-native-residue positions", "matched_delta")
    print("\nThe first line is the comparison that produced the original result. The")
    print("second holds native-residue identity fixed. If the first is positive and")
    print("the second is not, the effect was the selector excluding anchor-native")
    print("sites while the baseline kept them -- not a structural property of the")
    print("feature. Per-residue strata are in the CSV (matched_n, matched_source).")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
