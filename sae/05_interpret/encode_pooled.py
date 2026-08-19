"""LEGACY / SUPERSEDED -- kept for reference, not part of the current pipeline.

Originally written to pool per-residue sparse codes to one vector per
protein, feeding a planned `probe.py` + `cluster_crosscheck.py` split. Those
two scripts were never built -- `feature_analysis.py` (../05_interpret/)
now does its own pooling + linear probe inline instead. Nothing in the
documented pipeline (see ../README.md) calls this file; don't wire it into
a real run without first checking it still matches `data.py`/`sae_model.py`
(../03_train/), since it hasn't been exercised since that split happened.

Original docstring follows.

---

Pool per-residue sparse codes to one vector per protein, for every design
protein in a manifest (train+val both -- unlike benchmark.py, which only
scores the held-out split). Feeds `probe.py` and `cluster_crosscheck.py`
(never built -- see LEGACY note above).

Two modes, run separately (see each mode's docstring below for why):

    python encode_pooled.py ours --checkpoint <ckpt>.pt --data-dir <dir> \\
        --output pooled_codes_ours.npz
    python encode_pooled.py biohub --manifest <dir>/manifest_combined.csv \\
        --output pooled_codes_biohub.npz

Output: an .npz with `ids` (str array), `source` (str array), `mean_pooled`
and `max_pooled` (both (N, d_hidden) float16 -- dense, not sparse: mean-
pooling a nonnegative TopK code over a whole protein's residues means almost
every dimension that fired even once ends up nonzero, so a sparse format
buys little here and complicates the probe/cluster-crosscheck code that
reads this back in).
"""

import argparse
from pathlib import Path

# NOTE: import order matters on this machine -- numpy must load before torch,
# see sae/03_train/data.py's module docstring for the segfault this avoids.
import numpy as np
import pandas as pd
import torch


def run_ours(args) -> None:
    """Runs entirely locally: activations.npy/index.csv/manifest_combined.csv
    for the 65k design pool are already on disk (see sae/README.md), and our
    SAE is small enough to encode with on CPU -- no cluster/GPU needed."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_prepare_data"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))
    from data import center_scale  # noqa: E402
    from sae_model import SparseAutoencoder  # noqa: E402

    data_dir = Path(args.data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    manifest_df = pd.read_csv(data_dir / "manifest_combined.csv")[["id", "source"]]
    merged = index_df.merge(manifest_df, on="id", how="left")
    assert merged["source"].notna().all(), "some index.csv ids missing from manifest_combined.csv"

    if args.limit:
        merged = merged.iloc[: args.limit].reset_index(drop=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SparseAutoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean, scale = ckpt["mean"], ckpt["scale"]
    d_hidden = ckpt["config"]["d_hidden"]

    acts = np.load(data_dir / "activations.npy", mmap_mode="r")  # (total_residues, d_model)
    n = len(merged)
    mean_pooled = np.zeros((n, d_hidden), dtype=np.float16)
    max_pooled = np.zeros((n, d_hidden), dtype=np.float16)

    print(f"Encoding {n} proteins locally (dict_size={d_hidden})...")
    with torch.no_grad():
        for i, (start, length) in enumerate(zip(merged["start"], merged["length"])):
            residues = torch.from_numpy(np.asarray(acts[start : start + length]).copy())
            x_proc = center_scale(residues, mean, scale)
            latents = model.encode(x_proc)  # (length, d_hidden)
            mean_pooled[i] = latents.mean(dim=0).numpy().astype(np.float16)
            max_pooled[i] = latents.max(dim=0).values.numpy().astype(np.float16)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{n}")

    np.savez_compressed(
        args.output,
        ids=merged["id"].to_numpy(),
        source=merged["source"].to_numpy(),
        mean_pooled=mean_pooled,
        max_pooled=max_pooled,
    )
    print(f"Wrote {args.output} ({n} proteins, dict_size={d_hidden})")


def run_biohub(args) -> None:
    """MUST run on Waluigi inside /tmp/esm_verify_venv (not esmfold2_venv) --
    same constraint as benchmark.py, since this loads the real ESMC-300M
    model + Biohub's official SAE hook. Reuses benchmark.py's already-
    verified `run_batches` (same CLS/EOS masking) instead of re-deriving the
    encode path -- benchmark.py lives in ../04_benchmark/."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_benchmark"))
    from benchmark import BIOHUB_MODEL, BIOHUB_SAE_REPO, LAYER, run_batches  # noqa: E402
    from transformers import AutoModel, AutoTokenizer  # noqa: E402

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.iloc[: args.limit].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {BIOHUB_MODEL} + official SAE hook at layer {LAYER}...")
    model = AutoModel.from_pretrained(BIOHUB_MODEL, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(BIOHUB_MODEL)
    sae = AutoModel.from_pretrained(
        BIOHUB_SAE_REPO, allow_patterns=["config.json", f"layer_{LAYER}.safetensors"], device=model.device
    )
    sae.initialize_layers([LAYER])
    model.add_sae_models([sae.layers[str(LAYER)]])

    d_hidden = 16384
    sequences = manifest["sequence"].tolist()
    n = len(sequences)
    mean_pooled = np.zeros((n, d_hidden), dtype=np.float16)
    max_pooled = np.zeros((n, d_hidden), dtype=np.float16)

    print(f"Encoding {n} proteins via Biohub's official SAE...")
    done = 0
    for seq_i, _hidden, official_codes in run_batches(
        model, tokenizer, sequences, device, args.batch_size, smoke_test=False
    ):
        codes = official_codes.float()
        mean_pooled[seq_i] = codes.mean(dim=0).cpu().numpy().astype(np.float16)
        max_pooled[seq_i] = codes.max(dim=0).values.cpu().numpy().astype(np.float16)
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{n}")

    np.savez_compressed(
        args.output,
        ids=manifest["id"].to_numpy(),
        source=manifest["source"].to_numpy(),
        mean_pooled=mean_pooled,
        max_pooled=max_pooled,
    )
    print(f"Wrote {args.output} ({n} proteins, dict_size={d_hidden})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_ours = sub.add_parser("ours")
    p_ours.add_argument("--checkpoint", type=Path, required=True)
    p_ours.add_argument("--data-dir", type=Path, required=True)
    p_ours.add_argument("--output", type=Path, required=True)
    p_ours.add_argument("--limit", type=int, default=None, help="first N proteins only, for a quick sanity check")

    p_biohub = sub.add_parser("biohub")
    p_biohub.add_argument("--manifest", type=Path, required=True)
    p_biohub.add_argument("--output", type=Path, required=True)
    p_biohub.add_argument("--batch-size", type=int, default=16)
    p_biohub.add_argument("--limit", type=int, default=None, help="first N proteins only, for a quick sanity check")

    args = parser.parse_args()
    if args.mode == "ours":
        run_ours(args)
    else:
        run_biohub(args)


if __name__ == "__main__":
    main()
