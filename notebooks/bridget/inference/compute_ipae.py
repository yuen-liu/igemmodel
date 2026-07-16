"""Batch-compute ipAE (mean interface predicted aligned error) for a
directory of Boltz protein-design results.

This is NOT the same metric as Boltz's own `min_interaction_pae` (a minimum
over some interaction pairs). ipAE follows the BindCraft/ColabDesign
convention (colabdesign/af/loss.py::get_pae_loss): symmetrize the PAE matrix
((PAE + PAE.T) / 2), then take the mean over binder-row x target-column
residue pairs. We report the raw value in Angstroms (no /31 normalization --
BindCraft normalizes to 0-1 for its own AF2 pipeline, but a raw-A-scale
threshold like ipae<4 implies the unnormalized value).

Chain assignment (which token indices are target vs binder) is parsed from
the predicted .cif directly, using the same Boltz mmCIF field layout and
CA-token masking as the vendored ipsae.py (see vendor/ipsae.py), so results
are directly comparable to the ipSAE numbers computed there.

Usage:
    python compute_ipae.py --results-dir /path/to/vilip1_top500 --output ipae.csv
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CA_ATOM_NAMES = {"CA"}


def parse_cif_chains(cif_path: Path) -> np.ndarray:
    """Return an array of chain ids, one per CA-atom token, in file order --
    matching the token ordering boltz's pae.npz uses (see vendor/ipsae.py's
    Boltz mmCIF parsing logic, which this mirrors)."""
    field_idx = {}
    field_num = 0
    chains = []
    with open(cif_path) as f:
        for line in f:
            if line.startswith("_atom_site."):
                field_idx[line.strip().split(".")[1]] = field_num
                field_num += 1
                continue
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                atom_name = parts[field_idx["label_atom_id"]]
                seq_id = parts[field_idx["label_seq_id"]]
                if seq_id == ".":
                    continue  # ligand atom, not a CA token
                if atom_name in CA_ATOM_NAMES or "C1" in atom_name:
                    chain_id = parts[field_idx["auth_asym_id"]]
                    chains.append(chain_id)
    return np.array(chains)


def compute_one(result_dir: Path, target_chain: str, binder_chain: str):
    metadata_path = result_dir / "metadata.json"
    result_id = json.loads(metadata_path.read_text())["id"] if metadata_path.exists() else result_dir.name

    result_files = result_dir / "files" / "result"
    pae_path = result_files / "pae.npz"
    cif_candidates = list(result_files.glob("*_predicted.cif"))
    if not cif_candidates:
        return result_id, None, "no *_predicted.cif found"
    cif_path = cif_candidates[0]

    try:
        chains = parse_cif_chains(cif_path)
        pae_matrix = np.load(pae_path)["pae"]
        if pae_matrix.shape[0] != len(chains):
            return result_id, None, (
                f"pae matrix size {pae_matrix.shape[0]} != CA token count {len(chains)}"
            )

        p = (pae_matrix + pae_matrix.T) / 2.0
        binder_mask = chains == binder_chain
        target_mask = chains == target_chain
        if not binder_mask.any() or not target_mask.any():
            return result_id, None, f"missing chain {target_chain} or {binder_chain}"

        interface_block = p[np.ix_(binder_mask, target_mask)]
        ipae_value = float(interface_block.mean())
    except Exception as e:
        return result_id, None, f"{type(e).__name__}: {e}"

    return result_id, ipae_value, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="C")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    result_dirs = sorted(
        p.parent.parent.parent for p in args.results_dir.glob("**/files/result/pae.npz")
    )
    print(f"Found {len(result_dirs)} result folders under {args.results_dir}")
    if not result_dirs:
        raise ValueError("No result folders with files/result/pae.npz found")

    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(compute_one, rd, args.target_chain, args.binder_chain): rd
            for rd in result_dirs
        }
        done = 0
        for fut in as_completed(futures):
            result_id, ipae_value, error = fut.result()
            done += 1
            if error:
                errors.append((result_id, error))
            else:
                rows.append({"id": result_id, "ipae": ipae_value})
            if done % 2000 == 0 or done == len(result_dirs):
                print(f"  {done}/{len(result_dirs)} processed")

    if errors:
        print(f"{len(errors)} results failed, e.g.: {errors[:3]}")

    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} ipAE values to {args.output}")
    print(df["ipae"].describe())


if __name__ == "__main__":
    main()
