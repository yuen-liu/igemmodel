"""Batch-compute ipSAE (interface predicted aligned error score) for a
directory of Boltz protein-design results, using the vendored reference
implementation from DunbrackLab/IPSAE (vendor/ipsae.py, MIT license,
https://github.com/DunbrackLab/IPSAE, https://www.biorxiv.org/content/10.1101/2025.02.10.637595).

We call the vendored script verbatim as a subprocess per result rather than
reimplementing its math, to avoid introducing transcription bugs in a metric
we're using to filter real candidates. Its own PAE-cutoff-based scoring
(ipSAE_d0res, "max" variant -- the paper's recommended default) is parsed
out of the .txt summary it writes, then that file (plus its _byres.txt and
.pml siblings) is deleted to avoid littering the results tree at 20k scale.

Expects each result folder to contain files/result/pae.npz and
files/result/*_predicted.cif (the layout boltz-api's download-results
produces), and that the predicted complex has exactly a target chain and a
binder chain (default "A" and "C", matching this campaign's design spec).

Usage:
    python compute_ipsae.py --results-dir /path/to/vilip1_top500 --output ipsae.csv
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

VENDOR_SCRIPT = Path(__file__).parent / "vendor" / "ipsae.py"


def find_result_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        p.parent.parent.parent
        for p in results_dir.glob("**/files/result/pae.npz")
    )


def compute_one(result_dir: Path, target_chain: str, binder_chain: str, pae_cutoff: float, dist_cutoff: float):
    # Read the canonical result id from metadata.json rather than trusting the
    # folder name, which varies by export (e.g. "0001_site_pres_XXX" vs "pres_XXX").
    metadata_path = result_dir / "metadata.json"
    result_id = json.loads(metadata_path.read_text())["id"] if metadata_path.exists() else result_dir.name

    result_files = result_dir / "files" / "result"
    pae_path = (result_files / "pae.npz").resolve()
    cif_candidates = list(result_files.glob("*_predicted.cif"))
    if not cif_candidates:
        return result_id, None, "no *_predicted.cif found"
    cif_path = cif_candidates[0].resolve()

    proc = subprocess.run(
        [sys.executable, str(VENDOR_SCRIPT), str(pae_path), str(cif_path), str(pae_cutoff), str(dist_cutoff)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return result_id, None, f"ipsae.py failed: {proc.stderr.strip()[-500:]}"

    pae_string = str(int(pae_cutoff)).zfill(2)
    dist_string = str(int(dist_cutoff)).zfill(2)
    stem = cif_path.name.replace(".cif", "")
    out_txt = result_files / f"{stem}_{pae_string}_{dist_string}.txt"
    byres_txt = result_files / f"{stem}_{pae_string}_{dist_string}_byres.txt"
    pml = result_files / f"{stem}_{pae_string}_{dist_string}.pml"

    ipsae_value = None
    try:
        for line in out_txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            chn1, chn2, _pae, _dist, row_type = parts[0], parts[1], parts[2], parts[3], parts[4]
            if row_type != "max":
                continue
            if {chn1, chn2} == {target_chain, binder_chain}:
                ipsae_value = float(parts[5])
                break
    finally:
        for f in (out_txt, byres_txt, pml):
            f.unlink(missing_ok=True)

    if ipsae_value is None:
        return result_id, None, f"no 'max' row found for chains {target_chain}/{binder_chain}"
    return result_id, ipsae_value, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="C")
    parser.add_argument("--pae-cutoff", type=float, default=10.0)
    parser.add_argument("--dist-cutoff", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    result_dirs = find_result_dirs(args.results_dir)
    print(f"Found {len(result_dirs)} result folders under {args.results_dir}")
    if not result_dirs:
        raise ValueError("No result folders with files/result/pae.npz found")

    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                compute_one, rd, args.target_chain, args.binder_chain, args.pae_cutoff, args.dist_cutoff
            ): rd
            for rd in result_dirs
        }
        done = 0
        for fut in as_completed(futures):
            result_id, ipsae_value, error = fut.result()
            done += 1
            if error:
                errors.append((result_id, error))
            else:
                rows.append({"id": result_id, "ipsae": ipsae_value})
            if done % 50 == 0 or done == len(result_dirs):
                print(f"  {done}/{len(result_dirs)} processed")

    if errors:
        print(f"{len(errors)} results failed, e.g.: {errors[:3]}")

    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} ipSAE values to {args.output}")
    print(df["ipsae"].describe())


if __name__ == "__main__":
    main()
