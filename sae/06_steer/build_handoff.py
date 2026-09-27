"""Build the steering-candidate handoff CSV from injection sweep outputs.

Replaces the ad-hoc filtering that produced the first (buggy) version of
`steering_candidates_final.csv`. The qualifying condition is NOT
`aa_argmax_changed` -- that flag compares the pre-steering argmax to the
post-steering argmax, so it is True even when steering pulls the model's
pick back onto the native residue, which yields a `mutated_sequence`
identical to the input. Downstream (Boltz refold -> ProteinMPNN -> ESM-C
re-check) such a row is not a test of anything. Require instead that the
proposed residue actually differs from native.
"""

import argparse
import csv
from pathlib import Path

QUALIFY_COLUMNS = [
    "design_id", "feature_id", "resnum", "position", "native_aa",
    "argmax_aa_post", "alpha_multiplier", "mutated_sequence",
]


def is_true(value) -> bool:
    return str(value).strip().lower() == "true"


def qualifies(row, features) -> bool:
    return (
        row["feature_id"] in features
        and is_true(row["re_emerged"])
        and is_true(row["aa_argmax_changed"])
        # The condition the original handoff was missing.
        and row["native_aa"] != row["argmax_aa_post"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--features", nargs="+", default=["4657", "10586", "6073"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    features = set(args.features)
    rows = []
    for path in args.inputs:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    print(f"Read {len(rows)} injection rows from {len(args.inputs)} file(s)")

    kept, seen = [], set()
    for row in rows:
        if not qualifies(row, features):
            continue
        # One row per distinct proposed point mutation; the sweep repeats a
        # site across alpha multipliers. Keep the lowest alpha that worked.
        key = (row["design_id"], row["feature_id"], row["resnum"], row["argmax_aa_post"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    kept.sort(key=lambda r: (r["feature_id"], int(r["resnum"])))
    assert all(r["mutated_sequence"][int(r["position"])] == r["argmax_aa_post"] for r in kept)
    assert all(r["native_aa"] != r["argmax_aa_post"] for r in kept)

    out = Path(args.out)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALIFY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    per_feature = {}
    for row in kept:
        per_feature[row["feature_id"]] = per_feature.get(row["feature_id"], 0) + 1
    print(f"Wrote {len(kept)} sites to {out}")
    for feature in sorted(per_feature):
        print(f"  feature {feature}: {per_feature[feature]} sites")


if __name__ == "__main__":
    main()
