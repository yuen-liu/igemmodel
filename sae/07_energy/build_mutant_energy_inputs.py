"""Prepare the structural-arm folds for Prime interaction-energy profiling.

Closes the missing cell in the design matrix: the energy arm so far has only
ever run on *unmutated* designs (sae/07_energy/select_energy_sample.py, 477
originals), so every energy result is observational. The structural arm folded
100 structures for `233` -- 20 native + 20 steered mutant + 60 random-
substitution controls -- and those were scored for ProteinMPNN accommodation
and ipSAE, but never for interaction energy. Running the existing Prime
pipeline over them turns "designs carrying 233 bind better" into "installing
233's anchor changes binding energy by X".

Reads the fold list written by 06_steer/build_structural_handoff.py
(sae/results/run4/structural_folds_233.fasta, headers of the form
`<design_id>__native` / `__mutant_<nat><resnum><new>` / `__control_...`) and
writes, next to this script:

  mutant_energy_ids.txt                   one structure id per line
  interface_features_mutant_energy.yaml   a designs entry per structure id
  mutant_energy_manifest.csv              id -> design, arm, substitution

The YAML exists only because profile_binding_energy.py requires a designs
entry per id; the energy computation itself does not read it. `max_activation`
is deliberately null -- the SAE activation of a native design is not a property
of its mutants, and carrying it over would invite exactly the confusion this
arm is meant to resolve.

Usage:
    python build_mutant_energy_inputs.py --cif-dir <dir with the 100 folds>
    python build_mutant_energy_inputs.py --cif-dir folds/ --require-all
"""

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FASTA = REPO_ROOT / "sae/results/run4/structural_folds_233.fasta"
SITES = REPO_ROOT / "sae/results/run4/structural_sites_233.csv"

HEADER_RE = re.compile(
    r"^(?P<design_id>[^_]+(?:_[^_]+)*?)__(?P<arm>native|mutant|control)"
    r"(?:_(?P<native_aa>[A-Z])(?P<resnum>\d+)(?P<new_aa>[A-Z]))?$"
)


def parse_folds():
    if not FASTA.exists():
        raise SystemExit(f"{FASTA} not found -- run 06_steer/build_structural_handoff.py first.")
    rows = []
    for line in FASTA.read_text().splitlines():
        if not line.startswith(">"):
            continue
        name = line[1:].strip()
        m = HEADER_RE.match(name)
        if not m:
            raise SystemExit(f"unparsable FASTA header: {name!r}")
        g = m.groupdict()
        rows.append({
            "structure_id": name,
            "design_id": g["design_id"],
            "arm": g["arm"],
            "native_aa": g["native_aa"] or "",
            "resnum": int(g["resnum"]) if g["resnum"] else "",
            "new_aa": g["new_aa"] or "",
        })
    return rows


FULL20K = REPO_ROOT / "data/vilip1_full20k/results"


def stage_natives(rows, cif_dir):
    """Link the native backbones in from the local full-20k predictions.

    RESULTS.md: "All 20 native backbones are already on disk in
    data/vilip1_full20k/results/, so only the 80 mutant/control folds are
    strictly new work." Symlinked rather than copied -- these are inputs, not
    outputs, and there is no reason to duplicate them.
    """
    if not FULL20K.exists():
        print(f"  --stage-natives: {FULL20K} not present, skipping")
        return
    cif_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for r in rows:
        if r["arm"] != "native":
            continue
        dest = cif_dir / f"{r['structure_id']}.cif"
        if dest.exists() or dest.is_symlink():
            continue
        found = sorted((FULL20K / r["design_id"]).rglob("*.cif"))
        if not found:
            print(f"  --stage-natives: no CIF under {FULL20K / r['design_id']}")
            continue
        dest.symlink_to(found[0].resolve())
        staged += 1
    print(f"  --stage-natives: linked {staged} native CIF(s) into {cif_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cif-dir", type=Path, required=True,
                    help="Directory holding the folded structures, named <structure_id>.cif")
    ap.add_argument("--require-all", action="store_true",
                    help="Fail unless every fold in the FASTA has a CIF present.")
    ap.add_argument("--feature-id", default="233")
    ap.add_argument("--stage-natives", action="store_true",
                    help="Symlink the 20 native CIFs into --cif-dir from the local "
                         "full-20k predictions, so only the 80 mutant/control folds "
                         "have to be fetched.")
    args = ap.parse_args()

    rows = parse_folds()

    if args.stage_natives:
        stage_natives(rows, args.cif_dir)

    # Which folds actually exist. Andrew's naming may not match the FASTA
    # headers exactly, so report rather than silently emit a short list.
    present, missing = [], []
    for r in rows:
        cif = args.cif_dir / f"{r['structure_id']}.cif"
        (present if cif.exists() else missing).append(r)

    print(f"{len(rows)} folds in {FASTA.name}; {len(present)} CIFs found, {len(missing)} missing")
    if missing:
        print("  missing (first 5):", ", ".join(r["structure_id"] for r in missing[:5]))
        if args.require_all:
            raise SystemExit(
                "Refusing to emit a partial run with --require-all. If the CIFs are "
                "named differently, rename them to <structure_id>.cif or drop "
                "--require-all to profile only what is present."
            )
        if not present:
            raise SystemExit(f"No CIFs matched in {args.cif_dir} -- check the naming.")

    # Designs with an incomplete set cannot contribute a paired observation.
    by_design = {}
    for r in present:
        by_design.setdefault(r["design_id"], []).append(r)
    unpaired = [
        d for d, rs in by_design.items()
        if not ({"native", "mutant"} <= {x["arm"] for x in rs}
                and sum(x["arm"] == "control" for x in rs) >= 1)
    ]
    if unpaired:
        print(f"  WARNING: {len(unpaired)} design(s) lack a native+mutant+control set "
              f"and will drop out of the paired test: {', '.join(unpaired[:5])}")

    ids_path = HERE / "mutant_energy_ids.txt"
    ids_path.write_text("".join(f"{r['structure_id']}\n" for r in present))

    # Hand-written rather than via pyyaml: this must be readable by Schrodinger's
    # python, which has no libyaml, and the structure is trivial.
    yaml_path = HERE / "interface_features_mutant_energy.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Generated by build_mutant_energy_inputs.py -- metadata only.\n")
        f.write("# profile_binding_energy.py requires a designs entry per structure id;\n")
        f.write("# it does not read these values to compute the energy.\n")
        f.write("designs:\n")
        for r in present:
            f.write(f"  {r['structure_id']}:\n")
            f.write("    interface_features:\n")
            f.write(f"      '{args.feature_id}':\n")
            f.write(f"        tier: structural_arm_{r['arm']}\n")
            f.write("        max_activation: null\n")
        f.write("metadata:\n")
        f.write(f"  source_fasta: {FASTA.name}\n")
        f.write(f"  source_sites: {SITES.name}\n")
        f.write(f"  n_structures: {len(present)}\n")

    manifest = HERE / "mutant_energy_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure_id", "design_id", "arm", "native_aa", "resnum", "new_aa"])
        w.writeheader()
        w.writerows(present)

    print(f"wrote {ids_path}")
    print(f"wrote {yaml_path}")
    print(f"wrote {manifest}")
    print(f"\nNext: sbatch --array=0-2 run_mutant_energy_profiling.bash   # test first")
    print(f"      sbatch --array=0-{len(present) - 1} run_mutant_energy_profiling.bash")


if __name__ == "__main__":
    main()
