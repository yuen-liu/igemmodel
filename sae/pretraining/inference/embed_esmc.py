"""Embed designed binder sequences with ESM-C (biohub/ESMC-300M).

Standalone, GPU-batched embedding extraction -- deliberately does NOT load
binder_design.py's ESMFold2Design ensemble (inversion models + critics), which
is built for the design/optimization loop and is much heavier than plain
embedding extraction needs. This loads only the ESM-C language model.

Takes a manifest CSV (as written by rank_boltz_results.py, with at least
`id` and `sequence` columns) and writes a single .npz to --output.

Default (no --layers): single-layer mode, unchanged from the original
version of this script:
    ids: (N,) array of result IDs, same order as the manifest
    embeddings: (N, hidden_dim) float32 array, mean-pooled over residue
        positions (CLS/EOS/padding excluded), from the FINAL layer only.

With --layers (e.g. "--layers 15,21,final" or "--layers all"): multi-layer
mode. Still only ONE forward pass per batch -- output_hidden_states=True
already returns every layer, so pulling out extra layers costs no extra GPU
compute vs. the single-layer default, just a bit more pooling/RAM. We don't
know a priori how cluster structure varies with depth, so "all" extracts
every layer rather than guessing which ones matter. Output has:
    ids: (N,) array of result IDs
    layer_<i>: (N, hidden_dim) float32 array for each resolved layer index i
        (e.g. layer_0 ... layer_30), same pooling as above.

With --per-residue (requires exactly one resolved layer -- pass e.g.
"--layers 23"): skips pooling entirely and writes ONE activation vector per
residue instead of one per sequence, for training a sparse autoencoder (SAE)
on per-token ESM-C activations (see ../training/ for the SAE trainer that
consumes this output). Storing every layer per-residue would be ~30x the
single-layer size (infeasible at this corpus scale), so this mode
hard-errors if more than one layer is requested. --output must be a
directory in this mode; it's created (along
with parents) if missing, and gets two files instead of one .npz:
    activations.npy: (total_residues, hidden_dim) float16 memmap-backed
        array. Residues from all sequences are concatenated in manifest
        order (row i's sequence starts at index.csv's matching `start`),
        CLS/EOS/padding excluded -- same masking convention as the pooled
        path. float16 (not float32) because per-residue storage is ~L times
        bigger than pooled per-sequence storage (L = sequence length), and
        SAE training only needs reconstruction-level precision, not the
        exact bits.
    index.csv: id, start, length columns -- length is always equal to
        len(sequence) for that id (ESM-C's tokenizer is 1 token per
        residue), so activations.npy[start : start+length] recovers that
        id's residues in sequence order. total_residues is computed directly
        from manifest sequence lengths up front (no need to run the model
        first), so activations.npy is pre-sized as a disk-backed memmap
        instead of being concatenated in RAM at the end -- keeps peak memory
        well under the ~3-4GB full-corpus size regardless of batch size.

Layer indices follow the HF hidden_states convention: 0 = embedding output
(before any transformer block), i = output after transformer block i.
biohub/ESMC-300M has 30 transformer blocks, so hidden_states has 31 entries
(0..30); "final"/"last"/"-1" all resolve to 30 for this checkpoint, and
"all" resolves to 0..30. The actual layer count is detected at runtime from
the model's own first-batch output (not hardcoded), so this also works
unmodified against a different-sized ESM-C checkpoint passed via
--model-name.

IMPORTANT: run --smoke-test first and eyeball the printed shape/stats before
trusting a full run -- this hasn't been executed against the real ESMC
checkpoint yet, only written against the documented HF *ForMaskedLM contract
(output_hidden_states=True). If ESMCForMaskedLM's forward signature differs,
the smoke test is where that will surface, not partway through 500 sequences.

Usage:
    python embed_esmc.py --manifest manifest.csv --output embeddings.npz
    python embed_esmc.py --manifest manifest.csv --output /tmp/smoke.npz --smoke-test
    python embed_esmc.py --manifest manifest.csv --output embeddings_all_layers.npz --layers all
    python embed_esmc.py --manifest manifest.csv --output layer23_per_residue/ \\
        --layers 23 --per-residue
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
    layer_indices: list[int],
) -> dict[int, np.ndarray]:
    """Returns {layer_index: (B, hidden_dim) pooled embeddings}, one forward
    pass regardless of len(layer_indices)."""
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

    # Exclude CLS (position 0) and EOS (last real position per sequence) from
    # the mean so the embedding reflects residue composition, not boundary
    # tokens. attention_mask marks real tokens (including CLS/EOS) as 1.
    mask = attention_mask.clone().bool()
    seq_lens = attention_mask.sum(dim=1)  # includes CLS+EOS
    for i, length in enumerate(seq_lens.tolist()):
        mask[i, 0] = False  # CLS
        mask[i, length - 1] = False  # EOS
    mask_f = mask.unsqueeze(-1).float()
    denom = mask_f.sum(dim=1).clamp(min=1)

    pooled = {}
    for idx in layer_indices:
        hidden = outputs.hidden_states[idx].float()  # (B, L, D)
        pooled[idx] = ((hidden * mask_f).sum(dim=1) / denom).cpu().numpy()
    return pooled


def embed_batch_per_residue(
    model: ESMCForMaskedLM,
    tokenizer: ESMCTokenizer,
    sequences: list[str],
    device: torch.device,
    layer_idx: int,
) -> list[np.ndarray]:
    """Returns a list of (L_i, hidden_dim) float16 arrays, one per input
    sequence, CLS/EOS/padding excluded -- same per-example lengths as
    len(sequences[i]), same masking convention as embed_batch's pooling."""
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

    hidden = outputs.hidden_states[layer_idx].float()  # (B, L, D)
    seq_lens = attention_mask.sum(dim=1)  # includes CLS+EOS
    results = []
    for i, length in enumerate(seq_lens.tolist()):
        # Real residues are positions 1..length-2 (0 = CLS, length-1 = EOS).
        residues = hidden[i, 1 : length - 1].to(torch.float16).cpu().numpy()
        results.append(residues)
    return results


def run_per_residue(
    df: pd.DataFrame,
    tokenizer: ESMCTokenizer,
    model: ESMCForMaskedLM,
    device: torch.device,
    layers_arg: str,
    output_dir: Path,
    batch_size: int,
    smoke_test: bool,
) -> None:
    """Per-residue extraction path for SAE training data -- see the module
    docstring's --per-residue section for the on-disk format."""
    # Probe once up front (not lazily inside the batch loop, unlike the
    # pooled path) because we need hidden_dim before allocating the memmap,
    # and num_hidden_layers before resolving layers_arg.
    probe_encoded = tokenizer(df["sequence"].tolist()[:1], return_tensors="pt")
    with torch.no_grad():
        probe_out = model(
            input_ids=probe_encoded["input_ids"].to(device),
            attention_mask=probe_encoded["attention_mask"].to(device),
            output_hidden_states=True,
        )
    num_hidden_layers = len(probe_out.hidden_states) - 1
    hidden_dim = probe_out.hidden_states[0].shape[-1]
    del probe_out

    layer_indices = resolve_layer_indices(layers_arg, num_hidden_layers)
    if len(layer_indices) != 1:
        raise SystemExit(
            f"--per-residue requires exactly one resolved layer, got {len(layer_indices)} "
            f"from --layers {layers_arg!r} (model has {num_hidden_layers} transformer blocks)"
        )
    layer_idx = layer_indices[0]
    print(
        f"Model has {num_hidden_layers} transformer blocks, hidden_dim={hidden_dim}. "
        f"Extracting per-residue activations from layer {layer_idx}."
    )

    if smoke_test:
        sample = df["sequence"].tolist()[:2]
        results = embed_batch_per_residue(model, tokenizer, sample, device, layer_idx)
        for seq, arr in zip(sample, results):
            print(f"  seq len={len(seq)} -> activations shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")
        print("Smoke test passed (no crash, shapes look sane) -- not writing output.")
        return

    # Offsets/lengths are computed directly from manifest sequence lengths
    # (ESM-C tokenizes 1 token per residue), in ORIGINAL manifest order, so
    # activations.npy can be pre-sized and written to as a disk-backed
    # memmap -- no need to hold the whole corpus in RAM to concatenate at
    # the end the way the pooled path's np.stack does.
    lengths = df["sequence"].str.len().to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total_tokens = int(lengths.sum())
    print(f"Total residues across {len(df)} sequences: {total_tokens:,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path = output_dir / "activations.npy"
    activations = np.lib.format.open_memmap(
        activations_path, mode="w+", dtype=np.float16, shape=(total_tokens, hidden_dim)
    )

    # Sort by length to minimize padding waste in each batch (same trick as
    # the pooled path), but writes go to the ORIGINAL-order offsets computed
    # above, not sorted-batch order.
    order = df["sequence"].str.len().sort_values().index
    df_sorted = df.loc[order].reset_index(drop=False)  # "index" col = original row position

    start = time.time()
    for batch_start in range(0, len(df_sorted), batch_size):
        batch = df_sorted.iloc[batch_start : batch_start + batch_size]
        per_seq_acts = embed_batch_per_residue(model, tokenizer, batch["sequence"].tolist(), device, layer_idx)
        for orig_idx, acts in zip(batch["index"].tolist(), per_seq_acts):
            o = offsets[orig_idx]
            activations[o : o + len(acts)] = acts
        done = min(batch_start + batch_size, len(df_sorted))
        print(f"  {done}/{len(df_sorted)} sequences embedded")

    activations.flush()
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(df_sorted) / max(elapsed, 1e-9):.1f} seq/s)")
    print(
        f"Layer {layer_idx}: activations.npy shape={activations.shape} "
        f"mean={activations[:1000].astype(np.float32).mean():.4f} "
        f"(sampled from first 1000 rows)"
    )

    index_df = pd.DataFrame({"id": df["id"].to_numpy(), "start": offsets, "length": lengths})
    index_df.to_csv(output_dir / "index.csv", index=False)
    print(f"Wrote {activations_path} and {output_dir / 'index.csv'}")


def resolve_layer_indices(layers_arg: str, num_hidden_layers: int) -> list[int]:
    layers_arg = layers_arg.strip().lower()
    if layers_arg == "all":
        return list(range(0, num_hidden_layers + 1))

    indices = []
    for tok in layers_arg.split(","):
        tok = tok.strip().lower()
        if tok in ("final", "last", "-1"):
            idx = num_hidden_layers
        else:
            idx = int(tok)
            if idx < 0:
                idx = num_hidden_layers + 1 + idx
        if not (0 <= idx <= num_hidden_layers):
            raise ValueError(
                f"Layer index {idx} out of range for a model with "
                f"{num_hidden_layers} transformer blocks (valid: 0..{num_hidden_layers})"
            )
        indices.append(idx)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated hidden_states layer indices to extract (e.g. "
        "'15,21,final'), or 'all' for every layer. Omit for the original "
        "single-layer (final-layer-only) behavior.",
    )
    parser.add_argument(
        "--per-residue",
        action="store_true",
        help="Write one activation vector per residue instead of one "
        "mean-pooled vector per sequence (for SAE training). Requires "
        "--layers to resolve to exactly one layer; --output is treated as "
        "a directory (activations.npy + index.csv written inside it).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only embed the first 2 sequences and print diagnostics, then exit.",
    )
    args = parser.parse_args()
    multilayer = args.layers is not None

    if args.per_residue and not multilayer:
        raise SystemExit("--per-residue requires --layers to name exactly one layer (e.g. --layers 23)")

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

    if args.per_residue:
        run_per_residue(
            df, tokenizer, model, device, args.layers, args.output, args.batch_size, args.smoke_test
        )
        return

    # Sort by length to minimize padding waste in each batch, then restore
    # original order before saving.
    order = df["sequence"].str.len().sort_values().index
    df_sorted = df.loc[order].reset_index(drop=False)  # keep original index

    all_ids = df_sorted["id"].tolist()
    layer_indices: list[int] | None = None  # resolved lazily from first batch
    all_embeddings: dict[int, list] = {}

    start = time.time()
    for batch_start in range(0, len(df_sorted), args.batch_size):
        batch = df_sorted.iloc[batch_start : batch_start + args.batch_size]

        if layer_indices is None:
            # Probe hidden_states length on the first batch so "final"/"all"
            # resolve correctly without hardcoding the model's layer count.
            probe_encoded = tokenizer(batch["sequence"].tolist()[:1], return_tensors="pt")
            with torch.no_grad():
                probe_out = model(
                    input_ids=probe_encoded["input_ids"].to(device),
                    attention_mask=probe_encoded["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            num_hidden_layers = len(probe_out.hidden_states) - 1
            del probe_out
            if multilayer:
                layer_indices = resolve_layer_indices(args.layers, num_hidden_layers)
            else:
                layer_indices = [num_hidden_layers]
            print(
                f"Model has {num_hidden_layers} transformer blocks "
                f"({num_hidden_layers + 1} hidden_states entries). "
                f"Extracting {len(layer_indices)} layer(s): {layer_indices}"
            )
            all_embeddings = {idx: [None] * len(df_sorted) for idx in layer_indices}

        pooled = embed_batch(model, tokenizer, batch["sequence"].tolist(), device, layer_indices)
        for idx in layer_indices:
            for offset, emb in enumerate(pooled[idx]):
                all_embeddings[idx][batch_start + offset] = emb
        done = min(batch_start + args.batch_size, len(df_sorted))
        print(f"  {done}/{len(df_sorted)} sequences embedded")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(df_sorted) / max(elapsed, 1e-9):.1f} seq/s)")

    # Restore original manifest order before saving.
    restore_order = df_sorted["index"].values.argsort()
    ids_out = np.array(all_ids)[restore_order]

    layer_arrays = {}
    for idx in layer_indices:
        arr = np.stack(all_embeddings[idx])[restore_order]
        layer_arrays[idx] = arr
        print(f"Layer {idx}: shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")

    if args.smoke_test:
        print("Smoke test passed (no crash, shapes look sane) -- not writing output.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if multilayer:
        save_kwargs = {f"layer_{idx}": layer_arrays[idx] for idx in layer_indices}
        np.savez(args.output, ids=ids_out, **save_kwargs)
        print(
            f"Wrote {len(ids_out)} embeddings x {len(layer_indices)} layers "
            f"(keys: ids, {', '.join(f'layer_{i}' for i in layer_indices)}) to {args.output}"
        )
    else:
        np.savez(args.output, ids=ids_out, embeddings=layer_arrays[layer_indices[0]])
        print(f"Wrote {len(ids_out)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
