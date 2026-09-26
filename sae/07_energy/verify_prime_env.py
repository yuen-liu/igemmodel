"""Verify the Schrodinger Prime energy-profiling toolchain against ONE real
Boltz-predicted structure, before trusting a full batch run over ~550
designs. Mirrors sae/06_steer/inject_feature.py's --smoke-test philosophy:
no prior code in this repo or the fourier_loop_diffusion sibling repo does
PDB/mmCIF ingestion, chain-split interaction-energy scoring, or H-bond
detection with Schrodinger -- so every step here is a real, falsifiable
check, not an assumption.

What this checks, in order:
  1. Can schrodinger.structure.StructureReader read a Boltz .cif directly?
     (Everything in fourier_loop_diffusion assumes pre-existing .mae/.maegz
     files -- nothing there does PDB/mmCIF ingestion.)
  2. Chain-split interaction energy: minimize the full complex ONCE (the
     verified-working Prime.PrimeServer pattern from
     fourier_loop_diffusion/src/training/rl/scripts/prime_minimize.py),
     then take SINGLE-POINT energies of the target-alone and binder-alone
     structures extracted from that SAME minimized geometry (no separate
     re-minimization of the isolated partners). This measures interaction
     energy, not full binding free energy -- cheaper (one minimization per
     design, not three) and a defensible first-pass metric for ranking
     features against each other, but call it "interaction energy" (not
     "binding free energy") in any writeup.
  3. Hydrogen-bond detection -- no precedent anywhere, so this tries a
     couple of plausible Schrodinger API locations and prints what's
     actually there (dir() listing) if neither works cleanly, rather than
     silently guessing at a function name that might not exist.

Usage (on Gates, via the standard module-loaded Schrodinger install --
see sae/README.md or ask Bridget before touching `module unload`):
    $SCHRODINGER/run python verify_prime_env.py \\
        --structure /path/to/one/design/files/result/xxx_predicted.cif \\
        --target-chain A --binder-chain C
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", required=True, help="Path to one Boltz-predicted structure (.cif or .pdb)")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="C")
    args = parser.parse_args()

    print("=== Step 1: can StructureReader read this file directly? ===")
    try:
        from schrodinger.structure import StructureReader
    except ImportError as e:
        print(f"FAILED to import schrodinger.structure: {e}")
        print("Make sure you're running via $SCHRODINGER/run python, not system python.")
        sys.exit(1)

    try:
        st = StructureReader.read(args.structure)
    except Exception as e:
        print(f"FAILED to read {args.structure} directly: {type(e).__name__}: {e}")
        print("Fallback: convert with $SCHRODINGER/utilities/structconvert first, then retry with the .mae.")
        sys.exit(1)

    chain_names = sorted({c.name for c in st.chain})
    print(f"OK -- read {args.structure}")
    print(f"  title: {st.title!r}")
    print(f"  chains: {chain_names}")
    print(f"  total atoms: {len(list(st.atom))}")
    if args.target_chain not in chain_names or args.binder_chain not in chain_names:
        print(f"WARNING: expected chains {args.target_chain!r}/{args.binder_chain!r} not both found in "
              f"{chain_names} -- pass --target-chain/--binder-chain matching this structure's real chain ids.")
        sys.exit(1)

    print("\n=== Step 2: chain split + Prime interaction energy ===")
    from schrodinger.structutils.analyze import evaluate_asl
    from schrodinger.application.prime.packages import Prime
    import numpy as np

    all_atoms = np.array(evaluate_asl(st, "all"), dtype=np.int32)
    print(f"  minimizing full complex ({len(all_atoms)} atoms)...")
    with Prime.PrimeServer(st) as pes:
        pes.updateEnergyParams(include_pipack=False)
        pes.minimizeSelectedAtoms(all_atoms)
        e_complex = pes.calculateEnergy(st.getXYZ(), update_nb_list=1, update_sgb_opt=1)
    print(f"  E_complex (minimized) = {e_complex}")

    target_indices = evaluate_asl(st, f"chain.name {args.target_chain}")
    binder_indices = evaluate_asl(st, f"chain.name {args.binder_chain}")
    if not target_indices or not binder_indices:
        print(f"FAILED: chain.name ASL returned no atoms for target={args.target_chain!r} "
              f"({len(target_indices)} atoms) or binder={args.binder_chain!r} ({len(binder_indices)} atoms).")
        sys.exit(1)

    target_only = st.extract(target_indices)
    binder_only = st.extract(binder_indices)
    print(f"  target-only extract: {len(list(target_only.atom))} atoms")
    print(f"  binder-only extract: {len(list(binder_only.atom))} atoms")

    # Single-point energy on the geometry already relaxed as part of the
    # complex -- no separate minimization of the isolated partners. See
    # module docstring: this is "interaction energy", not full binding dG.
    with Prime.PrimeServer(target_only) as pes:
        pes.updateEnergyParams(include_pipack=False)
        e_target = pes.calculateEnergy(target_only.getXYZ(), update_nb_list=1, update_sgb_opt=1)
    with Prime.PrimeServer(binder_only) as pes:
        pes.updateEnergyParams(include_pipack=False)
        e_binder = pes.calculateEnergy(binder_only.getXYZ(), update_nb_list=1, update_sgb_opt=1)

    d_e_interaction = e_complex - e_target - e_binder
    print(f"  E_target (single-point) = {e_target}")
    print(f"  E_binder (single-point) = {e_binder}")
    print(f"  dE_interaction = E_complex - E_target - E_binder = {d_e_interaction}")
    if 1e8 in (e_complex, e_target, e_binder):
        print("WARNING: one of the energies hit 1e8 -- that's the existing failure-sentinel "
              "convention from fourier_loop_diffusion/src/training/rl/min_utils.py -- something "
              "went wrong in that particular minimization/energy call.")

    print("\n=== Step 3: hydrogen-bond detection (unverified API, exploratory) ===")
    try:
        from schrodinger.structutils import analyze as sd_analyze
        if hasattr(sd_analyze, "hbond"):
            print("  found schrodinger.structutils.analyze.hbond")
            print("  dir(analyze.hbond):", [n for n in dir(sd_analyze.hbond) if not n.startswith("_")])
        else:
            candidates = [n for n in dir(sd_analyze) if not n.startswith("_") and "bond" in n.lower()]
            print(f"  schrodinger.structutils.analyze has no 'hbond' attribute. "
                  f"bond-related names found: {candidates}")
    except Exception as e:
        print(f"  schrodinger.structutils.analyze import/inspection failed: {type(e).__name__}: {e}")

    try:
        from schrodinger.structutils.interactions import hbond as hbond_mod
        print("  found schrodinger.structutils.interactions.hbond module")
        print("  dir(hbond_mod):", [n for n in dir(hbond_mod) if not n.startswith("_")])
    except Exception as e:
        print(f"  schrodinger.structutils.interactions.hbond import failed: {type(e).__name__}: {e}")

    print("\nDone. Report back ALL printed output above -- especially the dir() listings if neither "
          "hbond candidate worked cleanly, so the real API can be pinned down before writing the "
          "batch script (sae/07_energy/profile_binding_energy.py, not yet written).")


if __name__ == "__main__":
    main()
