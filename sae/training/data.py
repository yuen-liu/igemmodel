"""Load ESM-C layer-23 per-residue activations and build a train/val split
for SAE training.

Expects a directory (e.g. vilip1_layer23_per_residue/, as written by
embed_esmc.py --per-residue) containing:
    activations.npy: (total_residues, d_model) float16, memmap-written.
    index.csv: id, start, length -- activations[start:start+length] is that
        id's residues, in sequence order.
    manifest_combined.csv: id, sequence, source -- source in
        {"vilip1_full20k", "composite_hotspot", "natural_binders"}.

`source == "natural_binders"` rows are dropped entirely here -- those 13
real UniProt sequences are held out as a qualitative eval-only set (not
statistically meaningful for training at 13/25013 rows, and structurally
very different: 84-815 residues vs. the two design campaigns' 50-80 and
100-150), evaluated separately, not part of this train/val split.

vilip1_full20k and composite_hotspot are the same POI (vilip1) but
different binder-length regimes (50-80 vs. 100-150 residues) at different
scale (20000 vs. 5000 proteins, but only 68/32 by RESIDUE count since
composite_hotspot's binders are ~2x longer -- less skewed than the protein
count alone suggests). The train/val split is stratified per source (each
source split 90/10 independently, not one pooled shuffle) so both splits
keep the same source mix, and `source` is carried alongside train_x/val_x
so validation metrics can be reported per source as well as pooled.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class SAEDataset:
    train_x: torch.Tensor  # (n_train, d_model) float16, RAW (uncentered, unscaled)
    val_x: torch.Tensor  # (n_val, d_model) float16, RAW
    train_source: np.ndarray  # (n_train,) str, row-aligned with train_x
    val_source: np.ndarray  # (n_val,) str, row-aligned with val_x
    mean: torch.Tensor  # (d_model,) float32, computed from train_x only
    scale: float  # computed from train_x only
    train_protein_ids: np.ndarray
    val_protein_ids: np.ndarray


def _stratified_protein_split(
    design_df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[set, set]:
    """Returns (train_ids, val_ids) sets of protein ids, split independently
    within each `source` group so both splits keep the same source mix."""
    rng = np.random.RandomState(seed)
    train_ids, val_ids = set(), set()
    for source in sorted(design_df["source"].unique()):
        ids = design_df.loc[design_df["source"] == source, "id"].to_numpy()
        perm = rng.permutation(len(ids))
        shuffled = ids[perm]
        n_val = int(round(len(ids) * val_fraction))
        val_ids.update(shuffled[:n_val].tolist())
        train_ids.update(shuffled[n_val:].tolist())
    return train_ids, val_ids


def _expand_to_residue_indices(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """rows: dataframe with start/length/source columns, one row per protein.
    Returns (residue_indices, residue_source) -- both length sum(rows.length),
    residue_source[i] is the source of the protein residue_indices[i] belongs to."""
    idx_parts = []
    source_parts = []
    for start, length, source in zip(rows["start"], rows["length"], rows["source"]):
        idx_parts.append(np.arange(start, start + length))
        source_parts.append(np.full(length, source, dtype=object))
    return np.concatenate(idx_parts), np.concatenate(source_parts)


def load_dataset(data_dir: Path, val_fraction: float = 0.1, seed: int = 0) -> SAEDataset:
    data_dir = Path(data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    manifest_df = pd.read_csv(data_dir / "manifest_combined.csv")[["id", "source"]]

    merged = index_df.merge(manifest_df, on="id", how="left")
    assert merged["source"].notna().all(), "some index.csv ids missing from manifest_combined.csv"

    design_df = merged[merged["source"] != "natural_binders"].reset_index(drop=True)
    print(
        f"{len(design_df)}/{len(merged)} proteins are designs (excluding "
        f"{len(merged) - len(design_df)} natural_binders, held out for eval only)"
    )

    train_ids, val_ids = _stratified_protein_split(design_df, val_fraction, seed)
    train_rows = design_df[design_df["id"].isin(train_ids)]
    val_rows = design_df[design_df["id"].isin(val_ids)]
    assert set(train_rows["id"]).isdisjoint(set(val_rows["id"]))

    train_idx, train_source = _expand_to_residue_indices(train_rows)
    val_idx, val_source = _expand_to_residue_indices(val_rows)

    for name, rows, idx in [("train", train_rows, train_idx), ("val", val_rows, val_idx)]:
        counts = rows["source"].value_counts()
        print(f"{name}: {len(rows)} proteins, {len(idx)} residues -- by source: {counts.to_dict()}")

    acts = np.load(data_dir / "activations.npy")  # full load into RAM, not memmap (~3.7GB total)
    train_x = torch.from_numpy(acts[train_idx].copy())
    val_x = torch.from_numpy(acts[val_idx].copy())
    del acts

    mean = train_x.float().mean(dim=0)
    centered_norms = (train_x.float() - mean).norm(dim=-1)
    scale = float(train_x.shape[1] ** 0.5 / centered_norms.mean().item())
    print(f"train mean norm={mean.norm().item():.4f}, scale={scale:.6f}")

    return SAEDataset(
        train_x=train_x,
        val_x=val_x,
        train_source=train_source,
        val_source=val_source,
        mean=mean,
        scale=scale,
        train_protein_ids=train_rows["id"].to_numpy(),
        val_protein_ids=val_rows["id"].to_numpy(),
    )


def iter_batches(x: torch.Tensor, batch_size: int, shuffle: bool = True, generator=None):
    """shuffle=True (training): random order, drops a ragged final batch.
    shuffle=False (validation): original row order preserved (so a parallel
    `source` array stays aligned), includes a final partial batch so every
    row is covered exactly once."""
    n = x.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        n_full = (n // batch_size) * batch_size
        for i in range(0, n_full, batch_size):
            yield x[perm[i : i + batch_size]]
    else:
        for i in range(0, n, batch_size):
            yield x[i : i + batch_size]


def center_scale(x: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.float() - mean) * scale


def uncenter_unscale(x_proc: torch.Tensor, mean: torch.Tensor, scale: float) -> torch.Tensor:
    return x_proc / scale + mean
