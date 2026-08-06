"""Benchmark our domain-specific vilip1 SAE against Biohub's OFFICIAL
general-purpose ESMC-300M SAE (layer 23) -- using Biohub's own encode path
(AutoModel + model.add_sae_models(...)), not a reverse-engineered formula.

Must run on the cluster, in a venv with the real `esm` package installed
(NOT esmfold2_venv -- installing `esm` fresh risks upgrading torch away
from the CUDA-11.8 pin that venv needs for Waluigi's older driver):

    source /tmp/esm_verify_venv/bin/activate
    # scp sae/training/{sae_model.py,data.py,benchmark.py}, our trained
    # checkpoint, and manifest_combined.csv up first
    python benchmark.py --checkpoint best.pt --manifest manifest_combined.csv \\
        --output-dir benchmark_results --smoke-test
    python benchmark.py --checkpoint best.pt --manifest manifest_combined.csv \\
        --output-dir benchmark_results

For each held-out validation sequence (protein ids stored in our own
checkpoint, from data.py's stratified split) and the eval-only natural-binder
sequences (data.EVAL_ONLY_SOURCES):
  1. Run the REAL biohub/ESMC-300M model with Biohub's OFFICIAL SAE hook
     attached at layer 23 (model.add_sae_models(...)) to get
     output["sae_outputs"]["layer23"] (their actual sparse codes) and
     output.hidden_states[23] (ground-truth per-residue activations,
     same layer/masking convention as embed_esmc.py).
  2. Decode Biohub's codes back to activation space via their published
     W_dec/b_dec (safetensors, downloaded once) -- a plain linear
     projection using their exact trained weights, not a guessed formula.
     Biohub's public API doesn't expose a decode() call, so this one step
     is unavoidable, but it does not touch their ENCODE side at all --
     that's 100% their official function.
  3. Run OUR trained SAE (from --checkpoint) on the SAME ground-truth
     hidden states.
  4. Compare FVE -- against a SHARED baseline (our checkpoint's train-set
     mean, not a naive zero baseline; raw-zero FVE is inflated here by
     ESM-C's large outlier activation dimensions, see project notes) --
     plus dead-feature fraction and per-source (vilip1_full20k vs.
     composite_hotspot) breakdown.
"""

import argparse
from pathlib import Path

import numpy  # import order guard -- see train.py's note (torch-before-numpy segfaults here)
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

from data import EVAL_ONLY_SOURCES, center_scale, uncenter_unscale
from sae_model import SparseAutoencoder

BIOHUB_MODEL = "biohub/ESMC-300M"
BIOHUB_SAE_REPO = "biohub/ESMC-300M-sae-k64-codebook16384"
LAYER = 23


def load_biohub_decoder_weights(layer: int = LAYER) -> tuple[torch.Tensor, torch.Tensor]:
    """W_dec/b_dec only -- decode is the one step Biohub's public API doesn't
    expose directly, so we apply their own published weights ourselves."""
    path = hf_hub_download(BIOHUB_SAE_REPO, f"layer_{layer}.safetensors")
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("W_dec"), f.get_tensor("b_dec")


class RunningStats:
    """Accumulates FVE/L0/dead-feature stats per (split, model, source)
    batch-by-batch, instead of holding every residue's full code vectors
    for an entire split in memory -- the latter OOM-kills on Waluigi once
    the held-out split reaches ~200k residues x Biohub's 16384-wide
    codebook (~13GB for that one tensor alone, on top of ours/x/recons)."""

    def __init__(self):
        self._entries: dict[tuple[str, str, str], dict] = {}

    def update(self, split: str, model: str, source: str, dict_size: int, x: torch.Tensor, recon: torch.Tensor, baseline_mean: torch.Tensor, codes: torch.Tensor) -> None:
        key = (split, model, source)
        e = self._entries.get(key)
        if e is None:
            e = {
                "dict_size": dict_size,
                "sq_err": 0.0,
                "sq_baseline": 0.0,
                "l0_sum": 0.0,
                "n": 0,
                "ever_nonzero": torch.zeros(dict_size, dtype=torch.bool, device=codes.device),
            }
            self._entries[key] = e
        e["sq_err"] += (x - recon).pow(2).sum().item()
        e["sq_baseline"] += (x - baseline_mean).pow(2).sum().item()
        e["l0_sum"] += (codes != 0).sum().item()
        e["n"] += x.shape[0]
        e["ever_nonzero"] |= (codes != 0).any(dim=0)

    def rows(self, split: str):
        for (s, model, source), e in self._entries.items():
            if s != split:
                continue
            yield {
                "split": split,
                "model": model,
                "source": source,
                "n_residues": e["n"],
                "fve": 1.0 - e["sq_err"] / max(e["sq_baseline"], 1e-12),
                "avg_l0": e["l0_sum"] / e["n"],
                "n_dead": e["dict_size"] - int(e["ever_nonzero"].sum().item()),
                "dict_size": e["dict_size"],
            }


def run_batches(model, tokenizer, sequences, device, batch_size, smoke_test):
    """Yields (hidden_layer23, official_sae_codes) per batch, both (N, D)
    with CLS/EOS/padding already excluded -- same masking convention as
    embed_esmc.py's embed_batch."""
    if smoke_test:
        sequences = sequences[:2]
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        batch_seqs = [sequences[i] for i in batch_idx]

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)

        hidden = output.hidden_states[LAYER].float()  # (B, L_padded, D)
        # sae_outputs is NOT (B, L_padded, codebook) like hidden_states -- it's
        # flat and padding-EXCLUDED: (sum of real per-sequence lengths, codebook).
        # Confirmed empirically: B=2, L_padded=79 would give 2*79=158 if padding
        # were included, but the real shape was 156 -- consistent with two real
        # (unpadded) lengths summing to 156, padding stripped entirely.
        official_flat = output["sae_outputs"][f"layer{LAYER}"]
        if official_flat.is_sparse:
            official_flat = official_flat.to_dense()
        official_flat = official_flat.float()  # (total_real_tokens_in_batch, codebook)

        attention_mask = inputs["attention_mask"]
        seq_lens = attention_mask.sum(dim=1)  # includes CLS+EOS, excludes padding
        cursor = 0
        for i, length in enumerate(seq_lens.tolist()):
            real = slice(1, length - 1)  # exclude CLS (0) and EOS (length-1)
            hidden_i = hidden[i, real]
            official_i = official_flat[cursor : cursor + length][1:-1]  # same CLS/EOS exclusion, flat-packed offsets
            cursor += length
            assert hidden_i.shape[0] == official_i.shape[0], (
                f"seq {i}: hidden gave {hidden_i.shape[0]} real residues, "
                f"official_flat slice gave {official_i.shape[0]} -- packing assumption is wrong"
            )
            yield batch_idx[i], hidden_i, official_i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # weights_only=False: safe here since this is our own checkpoint from our
    # own train.py run (contains numpy arrays for train/val protein ids,
    # which PyTorch 2.6+'s weights_only=True default blocks by default).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    our_model = SparseAutoencoder(**ckpt["config"]).to(device)
    our_model.load_state_dict(ckpt["model_state_dict"])
    our_model.eval()
    our_mean = ckpt["mean"].to(device)
    our_scale = ckpt["scale"]

    manifest = pd.read_csv(args.manifest)
    val_ids = set(ckpt["val_protein_ids"].tolist())
    val_rows = manifest[manifest["id"].isin(val_ids)]
    natural_rows = manifest[manifest["source"].isin(EVAL_ONLY_SOURCES)]

    print(f"Loading {BIOHUB_MODEL} + official SAE hook at layer {LAYER}...")
    model = AutoModel.from_pretrained(BIOHUB_MODEL, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(BIOHUB_MODEL)
    sae = AutoModel.from_pretrained(
        BIOHUB_SAE_REPO, allow_patterns=["config.json", f"layer_{LAYER}.safetensors"], device=model.device
    )
    sae.initialize_layers([LAYER])
    model.add_sae_models([sae.layers[str(LAYER)]])
    biohub_W_dec, biohub_b_dec = load_biohub_decoder_weights()
    biohub_W_dec, biohub_b_dec = biohub_W_dec.to(device), biohub_b_dec.to(device)

    our_dict_size = ckpt["config"]["d_hidden"]
    biohub_dict_size = 16384
    stats = RunningStats()
    for split_name, rows in [("held_out_designs", val_rows), ("natural_binders_qualitative", natural_rows)]:
        sequences = rows["sequence"].tolist()
        sources = rows["source"].tolist()
        ids = rows["id"].tolist()
        print(f"{split_name}: {len(sequences)} sequences" + (" (smoke test: first 2 only)" if args.smoke_test else ""))

        for seq_i, hidden, official_codes in run_batches(
            model, tokenizer, sequences, device, args.batch_size, args.smoke_test
        ):
            x_proc = center_scale(hidden, our_mean, our_scale)
            with torch.no_grad():
                our_recon_proc, _, _, our_codes = our_model(x_proc)
            our_recon = uncenter_unscale(our_recon_proc, our_mean, our_scale)

            # Biohub's real forward() (verified from their installed source,
            # transformers.models.esmc.modeling_esmc_sae._ESMCSAELayer)
            # z-score normalizes PER TOKEN before ever touching W_enc/W_dec/
            # b_dec, and returns/reconstructs entirely in that normalized
            # space -- it's never un-normalized internally. That normalization
            # has no learned parameters (just this token's own mean/std), so
            # we recompute the SAME per-token mean/std from the real ground-
            # truth `hidden` (same forward pass) and invert it ourselves to
            # compare reconstruction quality in raw activation-space units.
            token_mean = hidden.mean(dim=-1, keepdim=True)
            token_std = (hidden - token_mean).std(dim=-1, keepdim=True)
            biohub_recon_normalized = official_codes @ biohub_W_dec + biohub_b_dec
            biohub_recon = biohub_recon_normalized * (token_std + 1e-5) + token_mean

            source = sources[seq_i]
            # accumulate per-batch running stats only (not full per-residue
            # tensors for the whole split -- see RunningStats docstring)
            for src in ("__pooled__", source):
                stats.update(split_name, "ours", src, our_dict_size, hidden, our_recon, our_mean, our_codes)
                stats.update(split_name, "biohub", src, biohub_dict_size, hidden, biohub_recon, our_mean, official_codes)

            print(f"  {ids[seq_i]} (len={hidden.shape[0]}) processed")

        for name in ("ours", "biohub"):
            pooled = next(r for r in stats.rows(split_name) if r["model"] == name and r["source"] == "__pooled__")
            print(f"  {name}: fve={pooled['fve']:.4f} dead={pooled['n_dead']}/{pooled['dict_size']}")

    all_rows = [r for split_name in ("held_out_designs", "natural_binders_qualitative") for r in stats.rows(split_name)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    if args.smoke_test:
        print("\nSmoke test passed (no crash, numbers look sane) -- not writing output.")
        return
    out_df.to_csv(args.output_dir / "benchmark_summary.csv", index=False)
    print(f"\nWrote {args.output_dir / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
