"""Inject a candidate SAE feature's direction back into ESM-C and check
whether it causally re-fires, plus read the downstream sequence-level
(MLM logit) effect.

This is the "steering/ESM injections" step of the team's 4-person steering
pipeline (see sae/README.md's "Steering" section for the full mechanics
writeup this script implements): Vignesh identifies candidate features that
fire near the binder-target 3D interface with high delta-binding-energy
(extract_interface_features.py's output) -> this script injects each
candidate's direction back into the same ESM-C layer the SAE was trained on
and checks it causally re-fires -> Andrew re-predicts structure -> Andrew
runs ProteinMPNN inverse folding to check the effect survives at the
sequence level. Only the injection step is implemented here.

Mechanics, precisely:
    direction_raw = w_dec[feature_id] / scale
(w_dec lives in centered+scaled space; a *delta* added to raw activations
only needs unscaling, not un-centering -- mean cancels out for a
difference. See sae/README.md's "Mechanics" section for the derivation.)
A forward hook on the transformer block whose output equals
outputs.hidden_states[layer] adds `alpha * direction_raw` to that block's
output at one specific token position, then the forward pass continues
normally so the SAE re-encode and MLM-head logits both reflect the
intervention.

Where to inject: --interface-yaml (extract_interface_features.py's output)
is the only source of positions -- for each candidate feature, a sample of
designs where that feature already fires at the interface, restricted to
the exact resnums recorded there. Resnum -> sequence-position resolution
reuses extract_interface_features.py's own classify_design()/find_structures()
(hard sequence-verified, not arithmetic on PDB numbering) rather than
reimplementing that logic.

Two checks per (design, feature, position, alpha):
  - re-emergence: does the SAE code at that position come back nonzero/
    elevated after injection (re-encoding the steered hidden state)?
  - sequence-level: does the MLM head's amino-acid preference at that
    position shift between an unsteered and steered forward pass?

IMPORTANT -- unverified against the real model in this dev environment
(transformers.models.esmc is not installed here; only usable on
Waluigi/wherever ESM-C actually runs). Two things specifically need
confirming on real infra before trusting a full run:
  1. Block-index alignment: hidden_states[layer] is assumed to be the
     output of the (0-indexed) transformer block `layer - 1`.
  2. The low-level `model.esmc.transformer(..., layers_to_collect=[...])`
     path (seen in use at notebooks/bridget/binder_design.py:771-777) is
     used as a second, independent cross-check on (1) where possible, but
     its exact return signature for `layers_to_collect` is not documented
     anywhere in this repo -- verify_hook_alignment() degrades to a
     printed SKIPPED note (not a hard failure) if that path's return shape
     doesn't match what's expected, since only the primary hook-vs-
     hidden_states check is load-bearing for correctness.
--smoke-test runs both checks plus one synthetic steering pass on a
couple of sequences and writes nothing, mirroring embed_esmc.py's own
--smoke-test convention for the same reason: this hasn't been run against
the real checkpoint yet.

Usage:
    python inject_feature.py --smoke-test \\
        --checkpoint ../checkpoints/best.pt --data-dir ../data-dir

    python inject_feature.py \\
        --checkpoint ../checkpoints/best.pt --data-dir ../data-dir \\
        --interface-yaml ../05_interpret/interface_features.yaml \\
        --structures-dir /path/to/boltz_results \\
        --feature-stats-csv ../results/run4/feature_stats.csv \\
        --features 233,1707,995 \\
        --output injection_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401 -- must import before torch, see data.py's import-order note
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_prepare_data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_interpret"))
from data import center_scale  # noqa: E402
from feature_analysis import load_pool, load_sae  # noqa: E402
from extract_interface_features import classify_design, find_structures  # noqa: E402

DEFAULT_MODEL = "biohub/ESMC-300M"
# biohub/ESMC-300M's "main" ref moved to a different, incompatible checkpoint
# format (Llama-style key names: esmc.layers.N.self_attn.q_proj.weight, etc.)
# at some point after this transformers.models.esmc integration was written
# against the OLDER key naming (esmc.transformer.blocks.N.attn.k_ln.weight,
# etc.) -- loading "main" silently falls back to random init for every
# weight (confirmed 2026-09-26: all 80 blocks + norm + lm_head "not
# initialized", NaN by layer 1) rather than erroring. Pin the older,
# compatible revision explicitly rather than trusting "main".
DEFAULT_MODEL_REVISION = "a59b831785f907e96e6a246b1d142bfb76df31ee"
CANONICAL_AA_VOCAB_SLICE = slice(4, 24)  # notebooks/bridget/binder_design.py's convention


# ---------------------------------------------------------------------------
# Feature/alpha resolution
# ---------------------------------------------------------------------------


def resolve_features(args: argparse.Namespace, features_checked: list[int]) -> list[int]:
    """Same --features/--candidates-csv convention as extract_interface_features.py
    and label_features.py, plus: every requested id must already have interface
    positions recorded in --interface-yaml (there is nothing to inject at
    otherwise -- the fix is rerunning extract_interface_features.py for it,
    not guessing a position here)."""
    if args.features:
        requested = [int(f.strip()) for f in args.features.split(",")]
    elif args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        requested = candidates_df["feature"].astype(int).tolist()
    else:
        requested = list(features_checked)

    checked_set = set(features_checked)
    missing = [f for f in requested if f not in checked_set]
    if missing:
        raise ValueError(
            f"Feature id(s) {missing} are not in --interface-yaml's features_checked "
            f"({sorted(checked_set)}) -- rerun extract_interface_features.py including "
            f"them before injecting."
        )
    return requested


def load_feature_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "feature" in df.columns, f"{path} has no 'feature' column"
    return df.set_index("feature")


def get_alphas(feature_id: int, feature_stats: pd.DataFrame, args: argparse.Namespace) -> list[tuple[float, float]]:
    """Returns [(alpha_multiplier, alpha), ...]. alpha_multiplier is NaN when
    --alpha-absolute overrides feature_stats.csv entirely."""
    if args.alpha_absolute is not None:
        return [(float("nan"), args.alpha_absolute)]

    if feature_id not in feature_stats.index:
        raise ValueError(f"Feature {feature_id} not found in --feature-stats-csv")
    row = feature_stats.loc[feature_id]
    if bool(row["dead"]):
        raise ValueError(f"Feature {feature_id} is marked dead in --feature-stats-csv -- pick a live feature")
    mean_active = float(row["mean_activation_when_active"])
    if not np.isfinite(mean_active):
        raise ValueError(f"Feature {feature_id}'s mean_activation_when_active is not finite ({mean_active})")

    multipliers = [float(m.strip()) for m in args.alpha_multipliers.split(",")]
    return [(m, m * mean_active) for m in multipliers]


# ---------------------------------------------------------------------------
# Design/position selection -- --interface-yaml is the only source of
# injection targets (see module docstring).
# ---------------------------------------------------------------------------


def select_design_feature_pairs(
    yaml_data: dict, feature_ids: list[int], designs_per_feature: int, sampling: str, seed: int,
) -> dict[int, list[dict]]:
    """Returns {feature_id: [{"design_id", "tier", "contact": [...], "shell": [...],
    "max_activation"}, ...]} -- one entry per selected design for that feature,
    contact-tier designs preferred (sorted by max_activation desc for
    --sampling top, shuffled for --sampling random), falling back to
    shell-tier to fill the quota."""
    rng = np.random.default_rng(seed)
    selected: dict[int, list[dict]] = {}

    for feature_id in feature_ids:
        candidates = []
        for design_id, entry in yaml_data["designs"].items():
            feat_entry = entry.get("interface_features", {}).get(feature_id)
            if feat_entry is None:
                continue
            candidates.append({
                "design_id": design_id,
                "tier": feat_entry["tier"],
                "contact": feat_entry["residues"].get("contact", []),
                "shell": feat_entry["residues"].get("shell", []),
                "max_activation": feat_entry["max_activation"],
            })

        contact_pool = [c for c in candidates if c["tier"] == "contact"]
        shell_pool = [c for c in candidates if c["tier"] != "contact"]
        if sampling == "top":
            contact_pool.sort(key=lambda c: c["max_activation"], reverse=True)
            shell_pool.sort(key=lambda c: c["max_activation"], reverse=True)
        else:
            rng.shuffle(contact_pool)
            rng.shuffle(shell_pool)

        chosen = contact_pool[:designs_per_feature]
        n_remaining = designs_per_feature - len(chosen)
        if n_remaining > 0:
            chosen += shell_pool[:n_remaining]

        print(f"  feature {feature_id}: selected {len(chosen)} design(s) "
              f"({sum(c['tier'] == 'contact' for c in chosen)} contact-tier, "
              f"{sum(c['tier'] != 'contact' for c in chosen)} shell-tier fallback) "
              f"out of {len(candidates)} candidate(s)")
        selected[feature_id] = chosen

    return selected


def resolve_positions(
    design_id: str, structure_path: Path, expected_sequence: str,
    target_chain: str, binder_chain: str, contact_cutoff: float, shell_cutoff: float,
    resnums_wanted: list[int],
) -> dict[int, int]:
    """resnum -> 0-indexed sequence position, hard-verified via classify_design()
    (imported from extract_interface_features.py) rather than assuming PDB
    resnums are already 0-indexed positions."""
    _, _, resnums, error = classify_design(
        design_id, structure_path, expected_sequence,
        target_chain, binder_chain, contact_cutoff, shell_cutoff,
    )
    if error is not None:
        raise ValueError(f"{design_id}: could not resolve positions ({error})")
    positions = {}
    for resnum in resnums_wanted:
        if resnum not in resnums:
            raise ValueError(
                f"{design_id}: resnum {resnum} from --interface-yaml not found in "
                f"the re-parsed structure's resnums -- structures-dir may not match "
                f"the one used to build --interface-yaml"
            )
        positions[resnum] = resnums.index(resnum)
    return positions


# ---------------------------------------------------------------------------
# ESM-C model access -- transformer block enumeration + steering hook
# ---------------------------------------------------------------------------


def find_transformer_blocks(model) -> list[torch.nn.Module]:
    """Enumerate ESM-C's transformer blocks in order, by type rather than by
    guessing an attribute path -- same isinstance-based pattern already used
    in production in notebooks/bridget/binder_design.py (there, for
    activation-checkpointing; here, to find the injection point)."""
    from transformers.models.esmc.modeling_esmc import UnifiedTransformerBlock

    blocks = [m for m in model.modules() if isinstance(m, UnifiedTransformerBlock)]
    if not blocks:
        raise RuntimeError(
            "No UnifiedTransformerBlock instances found in the loaded model -- "
            "architecture may have changed since this script was written."
        )
    return blocks


def make_steering_hook(direction_raw: torch.Tensor, alpha: float, token_index: int):
    """Adds alpha * direction_raw to one token position in a transformer
    block's output. Batch size 1 only (position indexing doesn't account for
    padding across a batch -- see module docstring's batching rationale)."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output  # (1, L, D)
        hidden[0, token_index, :] += alpha * direction_raw
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    return hook


def embed_design(model, tokenizer, device: torch.device, sequence: str, layer: int, hook=None):
    """One tokenize + forward pass. Returns (hidden_layer: (L, D) float32 --
    real residues only, CLS/EOS excluded -- and logits: (L, vocab) float32,
    same slicing convention). hook, if given, is registered on the block
    producing hidden_states[layer] for the duration of this one forward pass."""
    blocks = find_transformer_blocks(model)
    if layer < 1 or layer > len(blocks):
        raise ValueError(f"--layer {layer} out of range for a {len(blocks)}-block model")
    target_block = blocks[layer - 1]  # hidden_states[0] = embeddings, hidden_states[i] = block i-1's output

    handle = target_block.register_forward_hook(hook) if hook is not None else None
    try:
        encoded = tokenizer([sequence], return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    finally:
        if handle is not None:
            handle.remove()

    length = int(attention_mask.sum().item())  # includes CLS+EOS
    hidden = outputs.hidden_states[layer][0, 1:length - 1].float().cpu()
    logits = outputs.logits[0, 1:length - 1].float().cpu()
    return hidden, logits


def verify_hook_alignment(model, tokenizer, device: torch.device, layer: int) -> bool:
    """Cross-checks the block-index assumption two ways before any real run
    trusts it: (1) a hook on blocks[layer-1] must reproduce
    outputs.hidden_states[layer] exactly (this is load-bearing -- hard
    failure aborts the run); (2) best-effort comparison against the
    low-level model.esmc.transformer(..., layers_to_collect=[layer]) path
    (notebooks/bridget/binder_design.py's usage) -- SKIPPED, not failed, if
    its return shape doesn't match what's expected, since that path's exact
    contract isn't documented anywhere in this repo."""
    blocks = find_transformer_blocks(model)
    target_block = blocks[layer - 1]

    captured = {}

    def capture_hook(module, inputs, output):
        captured["value"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    encoded = tokenizer([sequence], return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    handle = target_block.register_forward_hook(capture_hook)
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    finally:
        handle.remove()

    reference = outputs.hidden_states[layer].float()
    hooked = captured["value"].float()
    primary_ok = torch.allclose(reference, hooked, atol=1e-2, rtol=1e-2)
    print(f"  [verify] hook(blocks[{layer - 1}]) vs. hidden_states[{layer}]: "
          f"{'PASS' if primary_ok else 'FAIL'} "
          f"(max abs diff {(reference - hooked).abs().max().item():.4g})")

    if hasattr(outputs, "logits"):
        print(f"  [verify] outputs.logits shape: {tuple(outputs.logits.shape)} -- present, OK")
    else:
        print("  [verify] WARNING: outputs has no .logits attribute -- MLM readout will not work")
        primary_ok = False

    try:
        esmc = model.esmc
        embeds = esmc.embed(input_ids)
        result = esmc.transformer(embeds, sequence_id=None, layers_to_collect=[layer], output_attentions=False)
        extra = result[1:] if isinstance(result, tuple) else ()
        matched = any(
            isinstance(item, torch.Tensor) and item.shape == reference.shape
            and torch.allclose(item.float(), reference, atol=1e-2, rtol=1e-2)
            for item in extra
            if isinstance(item, torch.Tensor)
        )
        if matched:
            print(f"  [verify] low-level esmc.transformer(layers_to_collect=[{layer}]) path: PASS (matches)")
        else:
            print(f"  [verify] low-level esmc.transformer(layers_to_collect=[{layer}]) path: "
                  f"SKIPPED (returned {len(extra)} extra item(s), none matched hidden_states[{layer}]'s "
                  f"shape/values -- its exact return contract isn't documented; not treated as a failure)")
    except Exception as e:
        print(f"  [verify] low-level esmc.transformer(...) path: SKIPPED ({type(e).__name__}: {e})")

    return primary_ok


# ---------------------------------------------------------------------------
# Amino-acid logit summary
# ---------------------------------------------------------------------------


def build_canonical_aa_vocab(tokenizer) -> list[tuple[str, int]]:
    """The 20 canonical amino acids' (char, token_id), in the tokenizer's
    native order -- exact convention already used in
    notebooks/bridget/binder_design.py (lm_vocab[4:24])."""
    lm_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    return lm_vocab[CANONICAL_AA_VOCAB_SLICE]


def summarize_mlm_logits(logits_at_position: torch.Tensor, aa_vocab: list[tuple[str, int]], native_aa: str) -> dict:
    aa_chars = [c for c, _ in aa_vocab]
    aa_token_ids = [t for _, t in aa_vocab]
    restricted = logits_at_position[aa_token_ids]
    probs = torch.softmax(restricted, dim=-1)

    order = torch.argsort(probs, descending=True)
    ranked_chars = [aa_chars[i] for i in order.tolist()]
    ranked_probs = probs[order]

    argmax_aa = ranked_chars[0]
    argmax_prob = float(ranked_probs[0])
    if native_aa in aa_chars:
        native_idx = aa_chars.index(native_aa)
        native_rank = ranked_chars.index(native_aa)
        native_prob = float(probs[native_idx])
        native_logit = float(restricted[native_idx])
    else:
        native_rank, native_prob, native_logit = None, float("nan"), float("nan")

    return {
        "argmax_aa": argmax_aa, "argmax_prob": argmax_prob,
        "native_rank": native_rank, "native_prob": native_prob, "native_logit": native_logit,
    }


# ---------------------------------------------------------------------------
# Pool-side baseline (sanity check against a live forward pass)
# ---------------------------------------------------------------------------


def compute_pool_baseline_code(pool_df: pd.DataFrame, acts: np.ndarray, mean, scale, sae_model, device, design_id: str, feature_id: int, position: int) -> float:
    row = pool_df.loc[design_id]
    start, length = int(row["start"]), int(row["length"])
    x = torch.from_numpy(acts[start:start + length].astype(np.float32)).to(device)
    x_proc = center_scale(x, mean, scale)
    with torch.no_grad():
        code = sae_model.encode(x_proc)[position, feature_id].item()
    return code


# ---------------------------------------------------------------------------
# Per-row orchestration
# ---------------------------------------------------------------------------


def process_row(
    design_id, feature_id, resnum, position, residue_tier, design_max_activation,
    alpha_multiplier, alpha, mean_activation_when_active,
    sequence, native_aa, aa_vocab,
    model, tokenizer, device, layer, sae_model, mean, scale, direction_raw,
    pool_df, acts, pool_live_tolerance, reemergence_threshold,
) -> dict:
    token_index = position + 1

    pre_hidden, pre_logits = embed_design(model, tokenizer, device, sequence, layer, hook=None)
    pre_hidden_proc = center_scale(pre_hidden.to(device), mean, scale)
    with torch.no_grad():
        pre_code_live = sae_model.encode(pre_hidden_proc)[position, feature_id].item()
    pre_summary = summarize_mlm_logits(pre_logits[position], aa_vocab, native_aa)

    pre_code_pool = compute_pool_baseline_code(pool_df, acts, mean, scale, sae_model, device, design_id, feature_id, position)
    pre_pool_live_consistent = bool(
        abs(pre_code_pool - pre_code_live) <= pool_live_tolerance * max(abs(pre_code_live), 1e-6)
    )

    hook = make_steering_hook(direction_raw, alpha, token_index)
    post_hidden, post_logits = embed_design(model, tokenizer, device, sequence, layer, hook=hook)
    post_hidden_proc = center_scale(post_hidden.to(device), mean, scale)
    with torch.no_grad():
        post_code = sae_model.encode(post_hidden_proc)[position, feature_id].item()
    post_summary = summarize_mlm_logits(post_logits[position], aa_vocab, native_aa)

    return {
        "design_id": design_id, "feature_id": feature_id, "resnum": resnum, "position": position,
        "token_index": token_index, "residue_tier": residue_tier, "native_aa": native_aa,
        "design_max_activation": design_max_activation,
        "alpha_multiplier": alpha_multiplier, "mean_activation_when_active": mean_activation_when_active, "alpha": alpha,
        "pre_code_pool": pre_code_pool, "pre_code_live": pre_code_live,
        "pre_pool_live_consistent": pre_pool_live_consistent,
        "post_code": post_code, "code_delta": post_code - pre_code_live,
        "re_emerged": post_code >= reemergence_threshold * mean_activation_when_active if np.isfinite(mean_activation_when_active) else post_code > pre_code_live,
        "native_rank_pre": pre_summary["native_rank"], "native_prob_pre": pre_summary["native_prob"],
        "argmax_aa_pre": pre_summary["argmax_aa"], "argmax_prob_pre": pre_summary["argmax_prob"],
        "native_prob_post": post_summary["native_prob"],
        "argmax_aa_post": post_summary["argmax_aa"], "argmax_prob_post": post_summary["argmax_prob"],
        "aa_argmax_changed": pre_summary["argmax_aa"] != post_summary["argmax_aa"],
        "native_logit_shift": post_summary["native_logit"] - pre_summary["native_logit"],
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(args: argparse.Namespace) -> None:
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device}...")
    # dtype=torch.float32 (torch_dtype in older transformers) matches
    # notebooks/bridget/binder_design.py's loading convention -- without it,
    # from_pretrained may default to the checkpoint's stored dtype (bf16),
    # which is unreliable on CPU (NaN risk in attention/layernorm) since
    # there's no CUDA autocast managing it here.
    model = ESMCForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision, dtype=torch.float32
    ).to(device).eval()
    tokenizer = ESMCTokenizer()

    print("Verifying hook alignment against real infra (this is the point of --smoke-test)...")
    ok = verify_hook_alignment(model, tokenizer, device, args.layer)
    if not ok:
        print("Primary hook-vs-hidden_states check FAILED -- do not trust a real run until this passes.")
        return

    print("Loading SAE checkpoint for a synthetic steering pass...")
    sae_model, mean, scale, _ = load_sae(args.checkpoint, device)
    feature_stats = load_feature_stats(args.feature_stats_csv) if args.feature_stats_csv else None

    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTL"
    if feature_stats is not None and not feature_stats.empty:
        live_features = feature_stats[~feature_stats["dead"].astype(bool)]
        feature_id = int(live_features.index[0]) if not live_features.empty else int(feature_stats.index[0])
        mean_active = float(feature_stats.loc[feature_id, "mean_activation_when_active"])
    else:
        feature_id = 0
        mean_active = 5.0
    alpha = mean_active
    position = 1
    aa_vocab = build_canonical_aa_vocab(tokenizer)
    native_aa = sequence[position]

    direction_raw = (sae_model.w_dec[feature_id].detach() / scale).to(device)

    pre_hidden, pre_logits = embed_design(model, tokenizer, device, sequence, args.layer, hook=None)
    pre_code = sae_model.encode(center_scale(pre_hidden.to(device), mean, scale))[position, feature_id].item()
    pre_summary = summarize_mlm_logits(pre_logits[position], aa_vocab, native_aa)

    hook = make_steering_hook(direction_raw, alpha, position + 1)
    post_hidden, post_logits = embed_design(model, tokenizer, device, sequence, args.layer, hook=hook)
    post_code = sae_model.encode(center_scale(post_hidden.to(device), mean, scale))[position, feature_id].item()
    post_summary = summarize_mlm_logits(post_logits[position], aa_vocab, native_aa)

    print(f"\nSynthetic steering pass: feature {feature_id}, alpha={alpha:.4g}, position={position}")
    print(f"  SAE code: pre={pre_code:.4g} -> post={post_code:.4g}")
    print(f"  MLM argmax AA: pre={pre_summary['argmax_aa']} ({pre_summary['argmax_prob']:.3f}) "
          f"-> post={post_summary['argmax_aa']} ({post_summary['argmax_prob']:.3f})")
    print("\nNo output written (--smoke-test). If the code/argmax moved sensibly, proceed to a real run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="SAE checkpoint (required unless --smoke-test with no --feature-stats-csv)")
    parser.add_argument("--data-dir", type=Path, help="Matched activations.npy + index.csv + manifest_combined.csv pool")
    parser.add_argument("--interface-yaml", type=Path, help="extract_interface_features.py's output YAML")
    parser.add_argument("--structures-dir", type=Path, help="Same structures tree used to build --interface-yaml")
    parser.add_argument("--feature-stats-csv", type=Path, default=None, help="feature_analysis.py's feature_stats.csv (alpha source)")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature ids. Overrides --candidates-csv.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="A probe_*.csv or feature_labels.csv -- uses its 'feature' column.")
    parser.add_argument("--designs-per-feature", type=int, default=5)
    parser.add_argument("--sampling", choices=["top", "random"], default="top")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-multipliers", type=str, default="1.0", help="Comma-separated multipliers of mean_activation_when_active")
    parser.add_argument("--alpha-absolute", type=float, default=None, help="Override: use this exact alpha, ignoring --feature-stats-csv/--alpha-multipliers")
    parser.add_argument("--reemergence-threshold", type=float, default=0.5)
    parser.add_argument("--pool-live-tolerance", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=23, help="hidden_states index the SAE was trained on")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", type=str, default=DEFAULT_MODEL_REVISION,
                         help="Pinned HF revision -- 'main' has moved to an incompatible checkpoint format, see DEFAULT_MODEL_REVISION's comment")
    parser.add_argument("--smoke-test", action="store_true", help="Verify hook alignment + one synthetic pass; write nothing")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return

    required = ["checkpoint", "data_dir", "interface_yaml", "structures_dir", "feature_stats_csv", "output"]
    missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
    if missing:
        raise ValueError(f"Missing required argument(s) for a real run: {missing}")

    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_name} on {device}...")
    model = ESMCForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision, dtype=torch.float32
    ).to(device).eval()
    tokenizer = ESMCTokenizer()
    aa_vocab = build_canonical_aa_vocab(tokenizer)

    print("Verifying hook alignment before trusting any results from this run...")
    if not verify_hook_alignment(model, tokenizer, device, args.layer):
        raise RuntimeError("Hook-vs-hidden_states alignment check FAILED -- aborting before any injection.")

    print(f"Loading SAE checkpoint {args.checkpoint}...")
    sae_model, mean, scale, _ = load_sae(args.checkpoint, device)
    feature_stats = load_feature_stats(args.feature_stats_csv)

    print(f"Loading pool from {args.data_dir}...")
    pool_df = load_pool(args.data_dir).set_index("id")
    dup_ids = pool_df.index[pool_df.index.duplicated()].unique()
    if len(dup_ids):
        raise ValueError(f"--data-dir's manifest_combined.csv has duplicate id(s): {list(dup_ids[:3])}")
    acts = np.load(args.data_dir / "activations.npy", mmap_mode="r")

    print(f"Loading {args.interface_yaml}...")
    with open(args.interface_yaml) as f:
        yaml_data = yaml.safe_load(f)
    meta = yaml_data["metadata"]
    target_chain, binder_chain = meta["target_chain"], meta["binder_chain"]
    contact_cutoff, shell_cutoff = meta["contact_cutoff_angstrom"], meta["shell_cutoff_angstrom"]

    feature_ids = resolve_features(args, meta["features_checked"])
    print(f"Injecting {len(feature_ids)} feature(s): {feature_ids}")

    selected = select_design_feature_pairs(yaml_data, feature_ids, args.designs_per_feature, args.sampling, args.seed)

    structures_by_id = dict(find_structures(args.structures_dir))

    rows = []
    n_skipped = 0
    for feature_id in feature_ids:
        alphas = get_alphas(feature_id, feature_stats, args)
        direction_raw = (sae_model.w_dec[feature_id].detach() / scale).to(device)
        mean_active = float(feature_stats.loc[feature_id, "mean_activation_when_active"]) if feature_id in feature_stats.index else float("nan")

        for entry in selected[feature_id]:
            design_id = entry["design_id"]
            if design_id not in pool_df.index:
                print(f"  SKIP {design_id}/{feature_id}: not in --data-dir pool")
                n_skipped += 1
                continue
            if design_id not in structures_by_id:
                print(f"  SKIP {design_id}/{feature_id}: no structure found under --structures-dir")
                n_skipped += 1
                continue

            row = pool_df.loc[design_id]
            sequence = row["sequence"]
            resnums_wanted = sorted(set(entry["contact"]) | set(entry["shell"]))
            try:
                positions = resolve_positions(
                    design_id, structures_by_id[design_id], sequence,
                    target_chain, binder_chain, contact_cutoff, shell_cutoff, resnums_wanted,
                )
            except ValueError as e:
                print(f"  SKIP {design_id}/{feature_id}: {e}")
                n_skipped += 1
                continue

            for resnum, position in positions.items():
                residue_tier = "contact" if resnum in entry["contact"] else "shell"
                native_aa = sequence[position]
                for alpha_multiplier, alpha in alphas:
                    rows.append(process_row(
                        design_id, feature_id, resnum, position, residue_tier, entry["max_activation"],
                        alpha_multiplier, alpha, mean_active,
                        sequence, native_aa, aa_vocab,
                        model, tokenizer, device, args.layer, sae_model, mean, scale, direction_raw,
                        pool_df, acts, args.pool_live_tolerance, args.reemergence_threshold,
                    ))
                    if len(rows) % 50 == 0:
                        print(f"  ...{len(rows)} row(s) processed")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print(f"\nWrote {len(out_df)} row(s) to {args.output} ({n_skipped} design/feature pair(s) skipped)")
    if not out_df.empty:
        print(f"Re-emergence rate: {out_df['re_emerged'].mean():.1%}")
        n_inconsistent = (~out_df["pre_pool_live_consistent"]).sum()
        if n_inconsistent:
            print(f"WARNING: {n_inconsistent} row(s) had pre_code_pool/pre_code_live disagree beyond tolerance "
                  f"-- double-check the layer/CLS-offset convention before trusting results.")


if __name__ == "__main__":
    main()
