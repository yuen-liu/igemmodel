"""Merge multiple extract_interface_features.py output YAMLs (e.g. separate
per-batch runs over the same campaign/cutoffs) into one combined YAML.

Only safe to merge batches that used the IDENTICAL target/binder chain ids,
contact/shell cutoffs, and feature list -- mixing batches run with different
settings would silently combine incompatible tier definitions. This script
hard-fails rather than guessing if any input's metadata disagrees.

Usage:
    python merge_interface_yamls.py \\
        --inputs batch1.yaml batch2.yaml batch3.yaml \\
        --output interface_features_combined.yaml
"""

import argparse
from pathlib import Path

import yaml

# extract_interface_features.py's output is one entry per design and can run
# to hundreds of MB at full campaign scale (e.g. ~65k designs) -- PyYAML's
# pure-Python Loader/Dumper take tens of seconds per file at that size;
# libyaml's C bindings (CSafeLoader/CSafeDumper) are 5-10x faster. Fall back
# to the pure-Python versions only if libyaml isn't installed.
YAML_LOADER = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader
YAML_DUMPER = yaml.CSafeDumper if yaml.__with_libyaml__ else yaml.SafeDumper

METADATA_FIELDS_MUST_MATCH = [
    "target_chain", "binder_chain", "contact_cutoff_angstrom",
    "shell_cutoff_angstrom", "features_checked",
]


def merge(inputs: list[Path]) -> dict:
    loaded = []
    for path in inputs:
        with open(path) as f:
            loaded.append((path, yaml.load(f, Loader=YAML_LOADER)))

    reference_path, reference = loaded[0]
    reference_meta = reference["metadata"]
    for path, data in loaded[1:]:
        meta = data["metadata"]
        mismatched = [
            field for field in METADATA_FIELDS_MUST_MATCH
            if meta.get(field) != reference_meta.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"{path} disagrees with {reference_path} on {mismatched} -- "
                f"these batches were not run with the same settings and can't be safely merged. "
                f"{reference_path}: { {f: reference_meta.get(f) for f in mismatched} }, "
                f"{path}: { {f: meta.get(f) for f in mismatched} }"
            )

    merged_designs: dict = {}
    merged_skipped: dict = {}
    n_structures_found = 0
    for path, data in loaded:
        n_structures_found += data["metadata"].get("n_structures_found", 0)
        for design_id, entry in data.get("designs", {}).items():
            if design_id in merged_designs:
                raise ValueError(
                    f"design id {design_id!r} appears in more than one input file "
                    f"(at least {path} and an earlier one) -- batches were expected to "
                    f"cover disjoint designs; refusing to silently pick one."
                )
            merged_designs[design_id] = entry
        for design_id, reason in data.get("skipped", {}).items():
            if design_id in merged_skipped:
                continue  # same design skipped in more than one batch listing -- harmless, keep first reason
            merged_skipped[design_id] = reason

    output = {
        "metadata": {
            **{field: reference_meta[field] for field in METADATA_FIELDS_MUST_MATCH},
            "n_structures_found": n_structures_found,
            "n_designs_with_interface_features": len(merged_designs),
            "n_designs_skipped": len(merged_skipped),
            "merged_from": [str(p) for p, _ in loaded],
        },
        "designs": merged_designs,
        "skipped": merged_skipped,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Two or more extract_interface_features.py output YAMLs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Need at least 2 --inputs to merge")

    merged = merge(args.inputs)
    with open(args.output, "w") as f:
        yaml.dump(merged, f, Dumper=YAML_DUMPER, sort_keys=False, default_flow_style=False)

    print(f"Merged {len(args.inputs)} file(s): {merged['metadata']['n_designs_with_interface_features']} design(s), "
          f"{merged['metadata']['n_designs_skipped']} skipped")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
