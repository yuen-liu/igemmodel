"""Anchor-aware mutation selection from a top-k MLM readout.

The argmax readout deletes the residue a feature is defined on. Feature 4657
activates on serine in 15/15 of its top examples, yet injecting it makes the
MLM head prefer glutamate at the target position -- the direction encodes
"serine in acidic context" and a single-position argmax resolves it toward the
context, not the anchor. Every such proposal is self-defeating: it removes the
residue the feature requires (see RESULTS.md).

This walks the top-k ranking instead and applies the rule the argmax cannot:

  native is already an anchor residue  -> EXCLUDE. The anchor is present;
      no single-position substitution can improve the site, and every
      substitution destroys it. These sites need neighbour edits.
  an anchor residue appears in top-k   -> propose native -> that anchor,
      highest-ranked one. This installs what the feature detects.
  neither                              -> EXCLUDE. No anchor-consistent
      proposal exists at this position.

Anchor residues are derived from the bracketed position in
feature_top_examples.csv: the smallest set of residues covering
--anchor-coverage of a feature's top examples. A feature whose top examples
are spread across many residues gets a correspondingly wider anchor set,
which is the honest representation of a residue-class feature such as 10586
(hydrophobic) as against a single-residue one such as 4657 (serine).
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def derive_anchors(top_examples_csv: Path, coverage: float) -> dict[str, set[str]]:
    """feature_id -> set of anchor residues covering `coverage` of its examples."""
    per_feature: dict[str, Counter] = defaultdict(Counter)
    with open(top_examples_csv) as handle:
        for row in csv.DictReader(handle):
            context = row["context"]
            if "[" in context and "]" in context:
                per_feature[row["feature"]][context[context.index("[") + 1:context.index("]")]] += 1

    anchors = {}
    for feature, counts in per_feature.items():
        total = sum(counts.values())
        chosen, running = set(), 0
        for residue, count in counts.most_common():
            chosen.add(residue)
            running += count
            if running / total >= coverage:
                break
        anchors[feature] = chosen
    return anchors


def select(row: dict, anchors: set[str], exclude: frozenset[str] = frozenset()) -> tuple[str | None, str]:
    """Returns (proposed_aa, reason). proposed_aa is None when excluded."""
    native = row["native_aa"]
    if not anchors:
        return None, "no_anchor_derived"
    if native in anchors:
        return None, "native_is_anchor"
    blocked = False
    for candidate in row["top_k_aa_post"]:
        if candidate in anchors:
            if candidate in exclude:
                blocked = True
                continue
            return candidate, f"installs_anchor_rank{row['top_k_aa_post'].index(candidate) + 1}"
    return None, "anchor_excluded_by_chemistry" if blocked else "no_anchor_in_topk"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection-csv", type=Path, required=True,
                        help="inject_feature.py output including a top_k_aa_post column")
    parser.add_argument("--top-examples", type=Path, required=True)
    parser.add_argument("--anchor-coverage", type=float, default=0.6)
    parser.add_argument("--exclude-residues", type=str, default="",
                        help="Residues never to propose even when they are anchors, e.g. 'C' to avoid "
                             "installing cysteines (spurious disulfides / oxidation in a designed binder). "
                             "6073's anchor set is {C,L}, so C is reachable by default. Off unless set -- "
                             "this is a chemistry judgement, not something the ranking knows.")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated allowlist")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    exclude = frozenset(args.exclude_residues.strip().upper())
    anchors = derive_anchors(args.top_examples, args.anchor_coverage)
    if exclude:
        print(f"Excluding residues from proposals: {''.join(sorted(exclude))}")

    rows = list(csv.DictReader(open(args.injection_csv)))
    if "top_k_aa_post" not in rows[0]:
        raise SystemExit(
            f"{args.injection_csv} has no top_k_aa_post column -- it predates the top-k readout. "
            "Re-run inject_feature.py (--top-k defaults to 5) to produce it."
        )
    allow = set(args.features.split(",")) if args.features else None
    # Controls carry no mutation proposal; only the real feature direction does.
    rows = [r for r in rows
            if r.get("direction_mode", "feature") == "feature"
            and (allow is None or r["feature_id"] in allow)]

    print("Derived anchor sets (features present in this run):")
    for feature in sorted({r["feature_id"] for r in rows}, key=lambda f: (len(anchors.get(f, ())), f)):
        residues = "".join(sorted(anchors.get(feature, ()))) or "(none)"
        blocked = "".join(sorted(anchors.get(feature, set()) & exclude))
        note = f"   [excluded: {blocked}]" if blocked else ""
        print(f"  {feature:>6}: {residues}{note}")

    kept, reasons = [], Counter()
    for row in rows:
        proposed, reason = select(row, anchors.get(row["feature_id"], set()), exclude)
        reasons[reason] += 1
        if proposed is None:
            continue
        position = int(row["position"])
        native_seq = row["mutated_sequence"]
        # mutated_sequence carries the ARGMAX edit; undo it to recover native.
        native_seq = native_seq[:position] + row["native_aa"] + native_seq[position + 1:]
        kept.append({
            "design_id": row["design_id"], "feature_id": row["feature_id"],
            "resnum": row["resnum"], "position": position,
            "native_aa": row["native_aa"],
            "argmax_aa_post": row["argmax_aa_post"],
            "proposed_aa": proposed, "selection_reason": reason,
            "top_k_aa_post": row["top_k_aa_post"],
            "alpha_multiplier": row["alpha_multiplier"],
            "mutated_sequence": native_seq[:position] + proposed + native_seq[position + 1:],
        })

    seen, deduped = set(), []
    for row in sorted(kept, key=lambda r: (r["feature_id"], int(r["resnum"]), float(r["alpha_multiplier"]))):
        key = (row["design_id"], row["feature_id"], row["resnum"], row["proposed_aa"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    for row in deduped:
        assert row["mutated_sequence"][row["position"]] == row["proposed_aa"]
        assert row["proposed_aa"] != row["native_aa"]

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deduped[0]) if deduped else ["design_id"])
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nScreened {len(rows)} feature-direction rows:")
    for reason, count in reasons.most_common():
        print(f"  {count:>6}  {reason}")
    print(f"\nWrote {len(deduped)} distinct anchor-installing proposals to {args.out}")


if __name__ == "__main__":
    main()
