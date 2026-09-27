"""Compute Prime chain-split interaction energy + interface H-bonds for ONE
already-prepped design, and write its row(s) to a per-design CSV.

Verified in this repo's development (2026-09-26, sae/07_energy/verify_prime_env.py):
  - schrodinger.structure.StructureReader can read a prepwizard-prepped
    .mae directly (raw Boltz .cif fails Prime's Lewis-structure check --
    prepwizard must run first, see run_energy_profiling.bash).
  - Prime.PrimeServer chain-split interaction energy gives sane numbers
    (~-119 kcal/mol on the one test design) -- this is "interaction
    energy" (single minimization of the complex, then single-point energy
    of each isolated chain from that SAME geometry), not full binding
    free energy. See module docstring in verify_prime_env.py for why.
  - schrodinger.structutils.analyze.hbond.get_hydrogen_bonds exists and
    (per its docstring) returns a list of (donor_atom, acceptor_atom)
    tuples.

NOT yet verified with real data as of this writing: get_hydrogen_bonds'
actual output on a real complex (only the docstring was checked, not a
live call) -- test on a small --limit before trusting a full batch run.

One design can satisfy more than one candidate feature (see
sae/07_energy/select_energy_sample.py's "some designs may satisfy more
than one feature's sample" note) -- the expensive part (prep + energy +
hbond) runs ONCE per design; this script then writes one output row per
feature that design's interface-yaml entry lists, reusing the same energy
number for each.

Usage (one design; the SLURM array wrapper run_energy_profiling.bash calls
this once per design, --design-id resolved from the array index):
    $SCHRODINGER/run python3 profile_binding_energy.py \\
        --design-id pres_0A0NZAOnP7bs1ZD5CRnY \\
        --prepped-mae prepped/pres_0A0NZAOnP7bs1ZD5CRnY.mae \\
        --interface-yaml ../results/run4/yamls/interface_features_combined.yaml \\
        --target-chain A --binder-chain C \\
        --output-dir energy_results/
"""

import argparse
import sys
from pathlib import Path

import yaml

YAML_LOADER = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader


def compute_interaction_energy(st, target_chain: str, binder_chain: str):
    from schrodinger.structutils.analyze import evaluate_asl
    from schrodinger.application.prime.packages import Prime
    import numpy as np

    all_atoms = np.array(evaluate_asl(st, "all"), dtype=np.int32)
    with Prime.PrimeServer(st) as pes:
        pes.updateEnergyParams(include_pipack=False)
        pes.minimizeSelectedAtoms(all_atoms)
        e_complex = pes.calculateEnergy(st.getXYZ(), update_nb_list=1, update_sgb_opt=1)

    target_indices = evaluate_asl(st, f"chain.name {target_chain}")
    binder_indices = evaluate_asl(st, f"chain.name {binder_chain}")
    if not target_indices or not binder_indices:
        raise ValueError(f"chain.name ASL found no atoms for target={target_chain!r} or binder={binder_chain!r}")

    target_only = st.extract(target_indices)
    binder_only = st.extract(binder_indices)

    with Prime.PrimeServer(target_only) as pes:
        pes.updateEnergyParams(include_pipack=False)
        e_target = pes.calculateEnergy(target_only.getXYZ(), update_nb_list=1, update_sgb_opt=1)
    with Prime.PrimeServer(binder_only) as pes:
        pes.updateEnergyParams(include_pipack=False)
        e_binder = pes.calculateEnergy(binder_only.getXYZ(), update_nb_list=1, update_sgb_opt=1)

    return e_complex, e_target, e_binder


def compute_interface_hbonds(st, target_chain: str, binder_chain: str) -> int:
    """Count H-bonds crossing between target and binder chains -- i.e. at
    the interface, not internal to either chain."""
    from schrodinger.structutils.analyze import hbond

    bonds = hbond.get_hydrogen_bonds(st)
    n_interface = 0
    for donor, acceptor in bonds:
        chains = {donor.chain, acceptor.chain}
        if chains == {target_chain, binder_chain}:
            n_interface += 1
    return n_interface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--prepped-mae", type=Path, required=True, help="Output of prepwizard for this design")
    parser.add_argument("--interface-yaml", type=Path, required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="C")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.design_id}.csv"

    with open(args.interface_yaml) as f:
        yaml_data = yaml.load(f, Loader=YAML_LOADER)
    design_entry = yaml_data["designs"].get(args.design_id)
    if design_entry is None:
        print(f"ERROR: {args.design_id} not found in {args.interface_yaml}'s designs", file=sys.stderr)
        sys.exit(1)
    feature_entries = design_entry.get("interface_features", {})
    if not feature_entries:
        print(f"ERROR: {args.design_id} has no interface_features entries in the YAML", file=sys.stderr)
        sys.exit(1)

    from schrodinger.structure import StructureReader

    st = StructureReader.read(str(args.prepped_mae))
    e_complex, e_target, e_binder = compute_interaction_energy(st, args.target_chain, args.binder_chain)
    d_e_interaction = e_complex - e_target - e_binder
    n_interface_hbonds = compute_interface_hbonds(st, args.target_chain, args.binder_chain)

    rows = []
    for feature_id, feat in feature_entries.items():
        rows.append({
            "design_id": args.design_id,
            "feature_id": feature_id,
            "tier": feat["tier"],
            "max_activation": feat["max_activation"],
            "E_complex": e_complex,
            "E_target": e_target,
            "E_binder": e_binder,
            "dE_interaction": d_e_interaction,
            "n_interface_hbonds": n_interface_hbonds,
        })

    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{args.design_id}: dE_interaction={d_e_interaction:.2f} kcal/mol, "
          f"n_interface_hbonds={n_interface_hbonds}, {len(rows)} feature row(s) -> {out_path}")


if __name__ == "__main__":
    main()
