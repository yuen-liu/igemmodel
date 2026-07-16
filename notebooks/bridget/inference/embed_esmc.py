"""Embed designed binder sequences with ESM-C (biohub/ESMC-300M).

Standalone, GPU-batched embedding extraction -- deliberately does NOT load
binder_design.py's ESMFold2Design ensemble (inversion models + critics), which
is built for the design/optimization loop and is much heavier than plain
embedding extraction needs. This loads only the ESM-C language model.

Takes a manifest CSV (as written by rank_boltz_results.py, with at least
`id` and `sequence` columns) and writes a single .npz with:
    ids: (N,) array of result IDs, same order as the manifest
    embeddings: (N, hidden_dim) float32 array, mean-pooled over residue
        positions (CLS/EOS/padding excluded)

IMPORTANT: run --smoke-test first and eyeball the printed shape/stats before
trusting a full run -- this hasn't been executed against the real ESMC
checkpoint yet, only written against the documented HF *ForMaskedLM contract
(output_hidden_states=True). If ESMCForMaskedLM's forward signature differs,
the smoke test is where that will surface, not partway through 500 sequences.

Usage:
    python embed_esmc.py --manifest manifest.csv --output embeddings.npz
    python embed_esmc.py --manifest manifest.csv --output /tmp/smoke.npz --smoke-test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

DEFAULT_MODEL = "biohub/ESMC-300M"


def embed_batch(
    model: ESMCForMaskedLM,
    tokenizer: ESMCTokenizer,
    sequences: list[str],
    device: torch.device,
) -> np.ndarray:
    encoded = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[-1].float()  # (B, L, D)

    # Exclude CLS (position 0) and EOS (last real position per sequence) from
    # the mean so the embedding reflects residue composition, not boundary
    # tokens. attention_mask marks real tokens (including CLS/EOS) as 1.
    mask = attention_mask.clone().bool()
    seq_lens = attention_mask.sum(dim=1)  # includes CLS+EOS
    for i, length in enumerate(seq_lens.tolist()):
        mask[i, 0] = False  # CLS
        mask[i, length - 1] = False  # EOS

    mask_f = mask.unsqueeze(-1).float()
    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only embed the first 2 sequences and print diagnostics, then exit.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.smoke_test:
        df = df.head(2)
        print("Smoke test: embedding only the first 2 sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found, falling back to CPU (will be slow)")

    print(f"Loading {args.model_name}...")
    tokenizer = ESMCTokenizer()
    model = ESMCForMaskedLM.from_pretrained(args.model_name)
    model = model.to(device).eval().requires_grad_(False)

    # Sort by length to minimize padding waste in each batch, then restore
    # original order before saving.
    order = df["sequence"].str.len().sort_values().index
    df_sorted = df.loc[order].reset_index(drop=False)  # keep original index

    all_ids = df_sorted["id"].tolist()
    all_embeddings = [None] * len(df_sorted)

    start = time.time()
    for batch_start in range(0, len(df_sorted), args.batch_size):
        batch = df_sorted.iloc[batch_start : batch_start + args.batch_size]
        embeddings = embed_batch(model, tokenizer, batch["sequence"].tolist(), device)
        for offset, emb in enumerate(embeddings):
            all_embeddings[batch_start + offset] = emb
        done = min(batch_start + args.batch_size, len(df_sorted))
        print(f"  {done}/{len(df_sorted)} sequences embedded")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(df_sorted) / max(elapsed, 1e-9):.1f} seq/s)")

    embeddings_arr = np.stack(all_embeddings)
    print(f"Embedding matrix shape: {embeddings_arr.shape}")
    print(
        f"Per-dim stats: mean={embeddings_arr.mean():.4f} "
        f"std={embeddings_arr.std():.4f}"
    )

    # Restore original manifest order before saving.
    restore_order = df_sorted["index"].values.argsort()
    ids_out = np.array(all_ids)[restore_order]
    embeddings_out = embeddings_arr[restore_order]

    if args.smoke_test:
        print("Smoke test passed (no crash, shapes look sane) -- not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=ids_out, embeddings=embeddings_out)
    print(f"Wrote {len(ids_out)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
