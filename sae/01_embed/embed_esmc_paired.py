"""Embed binder sequences WITH the target (vilip1/VSNL1) present, at ESM-C
layer 23, writing out only the binder-span per-residue activations -- a
residue-aligned, drop-in alternative to embed_esmc.py's plain binder-alone
per-residue output, for the SAME protein ids. Step 2 of the SAE roadmap:
compare binder-alone vs. binder+target sparse-code activations/predictions
(sae/notebooks/, not yet extended for this -- see the plan).

Uses ESM-C's NATIVE chain-break token, `|` -- confirmed present in
`ESMCTokenizer().get_vocab()` (id 31, alongside the 20 amino acids and the
usual `<cls>`/`<eos>`/`<pad>`/`<mask>`/`<unk>`), and the same convention
already used elsewhere in this repo for multi-chain input
(`notebooks/bridget/binder_design.py`'s `TARGET_SEQUENCES` handling asserts
`"|" not in target_sequence` for single-chain targets, implying multi-chain
ones use it). Not a glycine-linker approximation -- this is the model's own
trained chain-break convention, so `binder + "|" + target` is a single
literal token sequence, not string concatenation glued together.

Target sequence: VSNL1/vilip1, full length (UniProt P62760):
https://www.uniprot.org/uniprotkb/P62760/entry#sequences

Usage:
    python embed_esmc_paired.py --manifest manifest.csv --output binder_plus_target_layer23_per_residue/ \\
        --layer 23 --smoke-test
    python embed_esmc_paired.py --manifest manifest.csv --output binder_plus_target_layer23_per_residue/ --layer 23
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

VSNL1_SEQUENCE = (
    "MGKQNSKLAPEVMEDLVKSTEFNEHELKQWYKGFLKDCPSGRLNLEEFQQLYVKFFPYGDASKFAQHAFRTFDKNGDGTIDFREFICALSITSRGSFEQK"
    "LNWAFNMYDLDGDGKITRVEMLEIIEAIYKMVGTVIMMKMNEDGLTPEQRVDKIFSKMDKNKDDQITLDEFKEAAKSDPSIVLLLQCDIQK"
)
DEFAULT_LINKER = "|"  # ESM-C's native chain-break token


def embed_batch_paired_per_residue(
    model: ESMCForMaskedLM,
    tokenizer: ESMCTokenizer,
    binder_sequences: list[str],
    target_sequence: str,
    linker: str,
    device: torch.device,
    layer_idx: int,
) -> list[np.ndarray]:
    """Returns a list of (L_i, hidden_dim) float16 arrays, one per binder --
    ONLY the binder-span residues (positions 1..len(binder)+1, right after
    CLS, since binder is placed first in the concatenation), even though the
    forward pass sees binder+linker+target together. Same CLS/EOS masking
    convention as embed_esmc.py's embed_batch_per_residue."""
    concatenated = [b + linker + target_sequence for b in binder_sequences]
    encoded = tokenizer(concatenated, return_tensors="pt", padding=True)
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

    hidden = outputs.hidden_states[layer_idx].float()  # (B, L_padded, D)
    results = []
    for i, binder_seq in enumerate(binder_sequences):
        # Real residues start at position 1 (0 = CLS); binder is first in
        # the concatenation, so its residues are exactly 1..len(binder_seq).
        binder_residues = hidden[i, 1 : 1 + len(binder_seq)].to(torch.float16).cpu().numpy()
        results.append(binder_residues)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with at least id,sequence columns (binder sequences)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (activations.npy + index.csv)")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--target-sequence", default=VSNL1_SEQUENCE, help="Override the target sequence (default: VSNL1/vilip1)")
    parser.add_argument("--linker", default=DEFAULT_LINKER, help="Sequence inserted between binder and target (default: ESM-C's native chain-break token, '|')")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true", help="Only embed the first 2 sequences and print diagnostics, then exit.")
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

    print(
        f"Target: {len(args.target_sequence)} residues, linker: {len(args.linker)} residues "
        f"-- concatenated as binder+linker+target, layer {args.layer}"
    )

    if args.smoke_test:
        sample = df["sequence"].tolist()
        results = embed_batch_paired_per_residue(model, tokenizer, sample, args.target_sequence, args.linker, device, args.layer)
        for seq, arr in zip(sample, results):
            print(f"  binder len={len(seq)} -> activations shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")
        print("Smoke test passed (no crash, shapes look sane) -- not writing output.")
        return

    # Probe hidden_dim from an actual forward pass rather than trusting a
    # config attribute name -- ESMCConfig doesn't expose `hidden_size`
    # (confirmed: AttributeError on a real run), same reasoning as
    # embed_esmc.py's run_per_residue.
    probe_encoded = tokenizer(df["sequence"].tolist()[:1], return_tensors="pt")
    with torch.no_grad():
        probe_out = model(
            input_ids=probe_encoded["input_ids"].to(device),
            attention_mask=probe_encoded["attention_mask"].to(device),
            output_hidden_states=True,
        )
    hidden_dim = probe_out.hidden_states[0].shape[-1]
    del probe_out
    print(f"hidden_dim={hidden_dim} (probed from a real forward pass)")

    lengths = df["sequence"].str.len().to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total_tokens = int(lengths.sum())
    print(f"Total binder residues across {len(df)} sequences: {total_tokens:,}")

    args.output.mkdir(parents=True, exist_ok=True)
    activations_path = args.output / "activations.npy"
    activations = np.lib.format.open_memmap(
        activations_path, mode="w+", dtype=np.float16, shape=(total_tokens, hidden_dim)
    )

    # Sort by BINDER length (the only thing that varies -- linker+target are
    # constant) to minimize padding waste; writes go to original-order offsets.
    order = df["sequence"].str.len().sort_values().index
    df_sorted = df.loc[order].reset_index(drop=False)

    start = time.time()
    for batch_start in range(0, len(df_sorted), args.batch_size):
        batch = df_sorted.iloc[batch_start : batch_start + args.batch_size]
        per_seq_acts = embed_batch_paired_per_residue(
            model, tokenizer, batch["sequence"].tolist(), args.target_sequence, args.linker, device, args.layer
        )
        for orig_idx, acts in zip(batch["index"].tolist(), per_seq_acts):
            o = offsets[orig_idx]
            activations[o : o + len(acts)] = acts
        done = min(batch_start + args.batch_size, len(df_sorted))
        print(f"  {done}/{len(df_sorted)} sequences embedded")

    activations.flush()
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(df_sorted) / max(elapsed, 1e-9):.1f} seq/s)")

    index_df = pd.DataFrame({"id": df["id"].to_numpy(), "start": offsets, "length": lengths})
    index_df.to_csv(args.output / "index.csv", index=False)
    print(f"Wrote {activations_path} and {args.output / 'index.csv'}")


if __name__ == "__main__":
    main()
