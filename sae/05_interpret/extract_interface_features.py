"""Map a chosen set of SAE features onto the binder-target structural
interface, per design.

Motivation: feature_analysis.py's max-activating examples are sequence-only
(local window around each firing residue), and label_features.py's LLM
labels are drafted from that same sequence-only evidence -- both explicitly
note this as a known limitation (see sae/README.md's step 6/7 and
label_features.py's module docstring): a feature whose real basis is
structural ("fires on interface-packing residues") can look like noise, or
worse a spurious sequence coincidence, without ever checking the actual 3D
structure. This script closes that gap for a specific question: of the
features we've already flagged as interesting (via the linear probe and/or
human labeling), which ones fire on residues that sit at or near the
binder-target interface -- i.e. plausibly binding-relevant, for better or
worse -- versus firing on residues buried in the core or on the solvent-
facing surface away from the target.

Interface definition: geometric, computed directly from each design's own
predicted 3D coordinates -- NOT the PAE-based criterion ipsae.py/ipSAE uses
internally (PAE < pae_cutoff selects residue pairs the model is CONFIDENT
about, which is an orthogonal axis to whether they're actually close in
space; a bad interface -- exactly the case we want features for -- is
likely to have high PAE right where it matters, so a confidence gate would
selectively exclude it). Instead: minimum heavy-atom distance between a
binder residue and any target-chain atom, the standard CAPRI/PISA contact
definition, with two tiers:
    contact: min distance <= --contact-cutoff  (default 5.0 A)
    shell:   contact-cutoff < min distance <= --shell-cutoff (default 10.0 A)
A residue farther than --shell-cutoff from the target is not reported.

Per-residue SAE codes are NOT recomputed from scratch here -- this assumes
the designs being analyzed are already part of the --data-dir activations
pool used to train/interpret the SAE (activations.npy + index.csv +
manifest_combined.csv, the same matched-pair data feature_analysis.py
consumes; see FEATURE_LABELING_SETUP.md). Re-embedding structures outside
that pool (fresh ESM-C forward pass) is out of scope here.

Position alignment (structure residue <-> activations.npy row) is done by
ORDER, not by trusting PDB/mmCIF residue numbers arithmetically (numbering
schemes vary), and then hard-verified: the binder chain's residues, taken
in ascending resnum order, must spell out exactly the same sequence as
manifest_combined.csv's `sequence` column for that id. Any mismatch (wrong
chain id, unexpected gap, id not in the pool) skips that design with a
recorded reason rather than silently misaligning residues to codes. The
resulting resnum -> position mapping is written into the output (see
`positions` below) so downstream consumers can act on a specific residue
without re-parsing the structure and redoing this same verification.

Usage:
    python extract_interface_features.py \\
        --checkpoint ../checkpoints/best.pt --data-dir ../data-dir \\
        --structures-dir /path/to/vilip1_full20k_results \\
        --features 233,1707,995 \\
        --output interface_features.yaml

    # or reuse a probe's candidate list instead of typing ids by hand:
    python extract_interface_features.py \\
        --checkpoint ../checkpoints/best.pt --data-dir ../data-dir \\
        --structures-dir /path/to/vilip1_full20k_results \\
        --candidates-csv ../results/probe_ipsae_max_multivariate.csv \\
        --output interface_features.yaml

Requires (beyond feature_analysis.py's own torch/numpy/pandas): pyyaml.
    pip install pyyaml
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Output can run to hundreds of MB at full campaign scale (~20k+ designs) --
# libyaml's C bindings (CSafeDumper) dump 5-10x faster than PyYAML's
# pure-Python SafeDumper. See merge_interface_yamls.py for the matching fix.
YAML_DUMPER = yaml.CSafeDumper if yaml.__with_libyaml__ else yaml.SafeDumper

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_prepare_data"))
from feature_analysis import load_pool, load_sae  # noqa: E402
from data import center_scale  # noqa: E402

CONTACT = "contact"
SHELL = "shell"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Structure parsing -- minimal PDB/mmCIF heavy-atom readers, same family of
# hand-rolled parsers already used by ipsae.py/compute_ipae.py in this repo
# (no biopython dependency). Ligand/hydrogen atoms are dropped; every other
# atom (including modified residues) is kept and grouped by (chain, resnum)
# at this parsing stage. A modified residue with no entry in THREE_TO_ONE
# maps to "X" for the sequence check below, which will not match the
# manifest's real one-letter code -- that design is then skipped downstream
# (see classify_design) as a sequence mismatch, not silently mishandled.
# ---------------------------------------------------------------------------


def parse_pdb_atoms(path: Path) -> list[dict]:
    atoms = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            if resname == "LIG":
                continue
            atom_name = line[12:16].strip()
            if atom_name[:1] == "H":
                continue
            atoms.append({
                "chain": line[21].strip(),
                "resnum": int(line[22:26]),
                "resname": resname,
                "atom_name": atom_name,
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
    return atoms


def parse_cif_atoms(path: Path) -> list[dict]:
    field_idx: dict[str, int] = {}
    field_num = 0
    atoms = []
    with open(path) as f:
        for line in f:
            if line.startswith("_atom_site."):
                field_idx[line.strip().split(".")[1]] = field_num
                field_num += 1
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            atom_name = parts[field_idx["label_atom_id"]]
            if atom_name[:1] == "H":
                continue
            seq_id = parts[field_idx["label_seq_id"]]
            if seq_id == ".":
                continue  # ligand atom, no residue number
            chain_field = "auth_asym_id" if "auth_asym_id" in field_idx else "label_asym_id"
            atoms.append({
                "chain": parts[field_idx[chain_field]],
                "resnum": int(seq_id),
                "resname": parts[field_idx["label_comp_id"]],
                "atom_name": atom_name,
                "x": float(parts[field_idx["Cartn_x"]]),
                "y": float(parts[field_idx["Cartn_y"]]),
                "z": float(parts[field_idx["Cartn_z"]]),
            })
    return atoms


def parse_structure(path: Path) -> list[dict]:
    if path.suffix == ".cif":
        return parse_cif_atoms(path)
    return parse_pdb_atoms(path)


# ---------------------------------------------------------------------------
# Structure discovery -- supports both the flat "one file per design" layout
# and the nested Boltz results-dir layout compute_ipsae.py/compute_ipae.py
# already read (**/files/result/*_predicted.cif + sibling metadata.json),
# so the same results folder used for ipSAE/ipAE can be reused directly.
# ---------------------------------------------------------------------------


def _dedupe_by_design_id(found: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Keep the first (sorted-order) path per design id; warn on collisions
    instead of letting a later result silently overwrite an earlier one."""
    by_id: dict[str, Path] = {}
    collisions: dict[str, int] = {}
    for design_id, path in found:
        if design_id in by_id:
            collisions[design_id] = collisions.get(design_id, 1) + 1
            continue
        by_id[design_id] = path
    if collisions:
        print(f"  WARNING: {len(collisions)} design id(s) matched more than one structure file; "
              f"keeping only the first, e.g.: {list(collisions.items())[:3]}")
    return list(by_id.items())


def find_structures(structures_dir: Path) -> list[tuple[str, Path]]:
    nested = sorted(structures_dir.glob("**/files/result/*_predicted.cif"))
    if nested:
        found = []
        for cif_path in nested:
            result_dir = cif_path.parent.parent.parent
            metadata_path = result_dir / "metadata.json"
            design_id = json.loads(metadata_path.read_text())["id"] if metadata_path.exists() else result_dir.name
            # Same convention as compute_ipsae.py/compute_ipae.py: if a result
            # dir has more than one *_predicted.cif, take the first (sorted).
            found.append((design_id, cif_path))
        return _dedupe_by_design_id(found)

    found = []
    for path in sorted(structures_dir.glob("**/*")):
        if path.suffix not in (".pdb", ".cif"):
            continue
        design_id = path.stem.replace("_predicted", "")
        found.append((design_id, path))
    return _dedupe_by_design_id(found)


# ---------------------------------------------------------------------------
# Interface classification (CPU/numpy only -- runs in worker processes)
# ---------------------------------------------------------------------------


def classify_design(
    design_id: str, structure_path: Path, expected_sequence: str,
    target_chain: str, binder_chain: str, contact_cutoff: float, shell_cutoff: float,
) -> tuple[str, list[str] | None, list[int] | None, str | None]:
    """Returns (design_id, tiers, resnums, error). tiers/resnums are parallel
    lists, one entry per binder residue in ascending-resnum order, aligned
    1:1 with `expected_sequence` (and therefore with activations.npy's rows
    for this design) -- or None/None with an error string if this design
    could not be safely aligned."""
    try:
        atoms = parse_structure(structure_path)
    except Exception as e:
        return design_id, None, None, f"failed to parse structure: {type(e).__name__}: {e}"

    try:
        target_atoms = np.array([[a["x"], a["y"], a["z"]] for a in atoms if a["chain"] == target_chain])
        if target_atoms.size == 0:
            return design_id, None, None, f"no atoms found for target chain {target_chain!r}"

        binder_residues: dict[int, dict] = {}
        for a in atoms:
            if a["chain"] != binder_chain:
                continue
            res = binder_residues.setdefault(a["resnum"], {"resname": a["resname"], "coords": []})
            res["coords"].append((a["x"], a["y"], a["z"]))
        if not binder_residues:
            return design_id, None, None, f"no atoms found for binder chain {binder_chain!r}"

        resnums = sorted(binder_residues)
        structure_seq = "".join(THREE_TO_ONE.get(binder_residues[r]["resname"], "X") for r in resnums)
        if not isinstance(expected_sequence, str) or structure_seq != expected_sequence:
            return design_id, None, None, (
                f"binder-chain sequence from structure ({len(structure_seq)} res) does not match "
                f"manifest sequence -- skipping to avoid misaligned residues"
            )

        # Vectorized: one (n_binder_atoms x n_target_atoms) distance matrix per
        # design instead of one small array per residue.
        atom_coords = []
        atom_res_idx = []
        for ri, r in enumerate(resnums):
            coords = binder_residues[r]["coords"]
            atom_coords.extend(coords)
            atom_res_idx.extend([ri] * len(coords))
        atom_coords = np.array(atom_coords)
        atom_res_idx = np.array(atom_res_idx)

        diffs = atom_coords[:, None, :] - target_atoms[None, :, :]
        atom_min_dist = np.sqrt((diffs ** 2).sum(-1)).min(axis=1)
        res_min_dist = np.full(len(resnums), np.inf)
        np.minimum.at(res_min_dist, atom_res_idx, atom_min_dist)

        tiers = []
        for min_dist in res_min_dist:
            if min_dist <= contact_cutoff:
                tiers.append(CONTACT)
            elif min_dist <= shell_cutoff:
                tiers.append(SHELL)
            else:
                tiers.append(None)

        return design_id, tiers, resnums, None
    except Exception as e:
        return design_id, None, None, f"failed to classify: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Feature-id resolution -- mirrors label_features.py's --features/
# --candidates-csv convention exactly, for consistency across the pipeline.
# ---------------------------------------------------------------------------


def resolve_features(args: argparse.Namespace) -> list[int]:
    if args.features:
        return [int(f.strip()) for f in args.features.split(",")]
    if args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        return candidates_df["feature"].astype(int).tolist()
    raise ValueError("Must pass --features or --candidates-csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Matched activations.npy + index.csv + manifest_combined.csv (same as feature_analysis.py).")
    parser.add_argument("--structures-dir", type=Path, required=True, help="Folder of predicted structures -- either flat *.pdb/*.cif files named by design id, or a Boltz results-dir tree (**/files/result/*_predicted.cif + metadata.json), same layout compute_ipsae.py reads.")
    parser.add_argument("--output", type=Path, required=True, help="Output YAML path.")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature ids. Overrides --candidates-csv.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="A probe_*_multivariate.csv/univariate.csv or feature_labels.csv -- uses its 'feature' column.")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="C")
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="Min heavy-atom distance (Angstrom) to call a binder residue a direct interface contact.")
    parser.add_argument("--shell-cutoff", type=float, default=10.0, help="Min heavy-atom distance (Angstrom) to call a binder residue interface-adjacent ('shell'), if not already a contact.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for structure parsing/classification (CPU-only stage).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_ids = resolve_features(args)
    print(f"Checking {len(feature_ids)} feature(s): {feature_ids}")

    model, mean, scale, _ = load_sae(args.checkpoint, device)
    pool_df = load_pool(args.data_dir).set_index("id")
    dup_ids = pool_df.index[pool_df.index.duplicated()].unique()
    if len(dup_ids):
        raise ValueError(
            f"--data-dir's manifest_combined.csv has {len(dup_ids)} duplicate id(s), e.g. "
            f"{list(dup_ids[:3])} -- fix the manifest before running (ids must be unique)."
        )
    acts = np.load(args.data_dir / "activations.npy", mmap_mode="r")

    structures = find_structures(args.structures_dir)
    print(f"Found {len(structures)} structure(s) under {args.structures_dir}")
    if not structures:
        raise ValueError(f"No .pdb/.cif structures found under {args.structures_dir}")

    designs_out: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    jobs = []
    for design_id, structure_path in structures:
        if design_id not in pool_df.index:
            skipped[design_id] = "id not found in --data-dir activations pool"
            continue
        jobs.append((design_id, structure_path, pool_df.loc[design_id, "sequence"]))

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                classify_design, design_id, structure_path, sequence,
                args.target_chain, args.binder_chain, args.contact_cutoff, args.shell_cutoff,
            ): design_id
            for design_id, structure_path, sequence in jobs
        }
        for fut in as_completed(futures):
            design_id, tiers, resnums, error = fut.result()
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"  classified {done}/{len(jobs)} structures")
            if error is not None:
                skipped[design_id] = error
                continue

            row = pool_df.loc[design_id]
            start, length = int(row["start"]), int(row["length"])
            x = torch.from_numpy(acts[start:start + length].astype(np.float32)).to(device)
            x_proc = center_scale(x, mean, scale)
            with torch.no_grad():
                codes = model.encode(x_proc)[:, feature_ids].cpu().numpy()  # (length, n_features)

            interface_features = {}
            for j, feature_id in enumerate(feature_ids):
                fired = np.nonzero(codes[:, j])[0]
                fired = [i for i in fired if tiers[i] is not None]
                if not fired:
                    continue
                contact_res = [resnums[i] for i in fired if tiers[i] == CONTACT]
                shell_res = [resnums[i] for i in fired if tiers[i] == SHELL]
                # resnum -> 0-indexed sequence/activations-array position, for
                # consumers (e.g. 06_steer/inject_feature.py) that need to act
                # on a specific residue without re-parsing the structure and
                # re-deriving this same already-hard-verified mapping.
                positions = {resnums[i]: int(i) for i in fired}
                interface_features[feature_id] = {
                    "tier": CONTACT if contact_res else SHELL,
                    "residues": {"contact": contact_res, "shell": shell_res},
                    "positions": positions,
                    "max_activation": float(codes[fired, j].max()),
                }

            if interface_features:
                designs_out[design_id] = {
                    "n_binder_residues": length,
                    "interface_features": interface_features,
                }

    output = {
        "metadata": {
            "target_chain": args.target_chain,
            "binder_chain": args.binder_chain,
            "contact_cutoff_angstrom": args.contact_cutoff,
            "shell_cutoff_angstrom": args.shell_cutoff,
            "features_checked": feature_ids,
            "n_structures_found": len(structures),
            "n_designs_with_interface_features": len(designs_out),
            "n_designs_skipped": len(skipped),
        },
        "designs": designs_out,
        "skipped": skipped,
    }
    with open(args.output, "w") as f:
        yaml.dump(output, f, Dumper=YAML_DUMPER, sort_keys=False, default_flow_style=False)

    print(f"\n{len(designs_out)}/{len(jobs)} designs had >=1 requested feature firing within {args.shell_cutoff}A of the interface")
    if skipped:
        print(f"{len(skipped)} designs skipped, e.g.: {list(skipped.items())[:3]}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
