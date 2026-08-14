"""Semantic / feature analysis for a trained protein SAE (see sae_model.py).

"Semantic analysis" here means: what does each of the SAE's d_hidden sparse
features actually respond to, and does that response mean anything
biologically? This is the standard SAE-interpretability recipe (Cunningham
et al. 2023; Bricken et al. 2023, "Towards Monosemanticity"), adapted from
language-model neurons to per-residue protein activations:

  1. Run the trained encoder over a big pool of real residues (here: ESM-C
     layer-23 activations from actual binder designs) and, for each of the
     d_hidden features, find the residues that make it fire HARDEST --
     "max activating examples". Reading the local sequence context around
     those residues is how a human assigns the feature a semantic label,
     e.g. "fires on leucines in buried helix interfaces" or "fires right
     after a proline kink". This is FEATURE_STATS + TOP EXAMPLES below.
  2. Track how often each feature fires at all (density). A feature active
     on <0.1% of residues is a specific pattern-detector; one active on 40%
     is closer to a generic property (or a hint that k=64 is being spread
     too thin). A feature that NEVER fires is dead (see sae_model.py's
     AuxK machinery, which tries to prevent this during training).
  3. Pool per-residue codes into one vector per protein (max-pool: does
     this feature fire ANYWHERE in this design) and check whether that
     pooled code predicts a downstream label the SAE was never trained on
     -- binding_confidence/iptm/ipsae/ipae. A feature that is BOTH
     interpretable (step 1) AND predictive (step 3) is the strongest
     evidence the SAE has found a real, usable structural correlate of
     binding, not just a reconstruction artifact. This is the LINEAR PROBE
     below -- see the SAE README's "Open next steps".

     

Design choice: max-activating-example search runs on a stratified RANDOM
SUBSAMPLE of residues (--max-residues, default 200k), not the full ~2M-
residue pool. Two reasons: (a) it keeps a Python-level streaming top-k scan
tractable (see TopActivations), and (b) it's standard practice in SAE
interpretability work (e.g. Anthropic's own feature dashboards) to draw
max-activating examples from a large-but-bounded pool rather than the
literal entire training set -- diminishing returns past a few hundred
thousand examples for a dictionary this size. The linear probe (step 3)
does NOT subsample: it needs each protein's COMPLETE pooled footprint, not
a sample of its residues, so it re-encodes every residue of every protein
being probed.

Run on the cluster, wherever --data-dir's activations.npy lives (same
requirement as train.py/benchmark.py):

    python feature_analysis.py --checkpoint best.pt \\
        --data-dir vilip1_layer23_per_residue_65k \\
        --output-dir feature_analysis_results

    # add a linear probe against a per-campaign manifest with metric columns
    # (produced by scripts/rank_boltz_results.py + compute_ipsae.py/
    # compute_ipae.py -- see notebooks/bridget/esmc_embedding_analysis/ for
    # how those get joined into one CSV):
    python feature_analysis.py --checkpoint best.pt \\
        --data-dir vilip1_layer23_per_residue_65k \\
        --output-dir feature_analysis_results \\
        --probe-metrics-csv vilip1_full20k/manifest_with_metrics.csv \\
        --probe-targets binding_confidence,iptm,ipsae,ipae

Writes (to --output-dir):
    feature_stats.csv          -- one row per feature: density, fire_count,
                                   mean_activation_when_active, dead.
    feature_top_examples.csv   -- top --top-n activating residues per
                                   feature: protein id, source, position,
                                   activation, local sequence context.
    probe_<target>_univariate.csv    -- per-feature Spearman rho vs. target
                                         (only written if --probe-metrics-csv given).
    probe_<target>_multivariate.csv  -- nonzero Lasso coefficients vs. target.
    probe_<target>_summary.json      -- n_proteins, cross-validated R^2, alpha.
"""

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

# NOTE: import order matters on this machine -- see train.py's note (torch
# before numpy segfaults). Import numpy-touching modules before torch.
import numpy as np  # noqa: F401
import pandas as pd
from data import EVAL_ONLY_SOURCES, center_scale
import torch

from sae_model import SparseAutoencoder


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_sae(checkpoint_path: Path, device: torch.device):
    """Loads our trained checkpoint the same way benchmark.py does. Returns
    (model, mean, scale, val_protein_ids) -- val_protein_ids is the held-out
    set from training's stratified split (data.py), the natural default
    population for the probe: never seen during training, so a predictive
    result there isn't just memorization."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SparseAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = ckpt["mean"].to(device)
    scale = ckpt["scale"]
    return model, mean, scale, ckpt["val_protein_ids"]


def load_pool(data_dir: Path) -> pd.DataFrame:
    """One row per protein: id, start, length, sequence, source. Unlike
    data.py's load_dataset, this does NOT drop EVAL_ONLY_SOURCES or split
    train/val -- feature interpretation wants the whole pool, including the
    real natural-binder proteins (arguably the most interesting case: do
    features found in synthetic designs also fire on real proteins?)."""
    data_dir = Path(data_dir)
    index_df = pd.read_csv(data_dir / "index.csv")
    manifest_df = pd.read_csv(data_dir / "manifest_combined.csv")
    merged = index_df.merge(manifest_df, on="id", how="left")
    assert merged["source"].notna().all(), "some index.csv ids missing from manifest_combined.csv"
    return merged


# ---------------------------------------------------------------------------
# Step 1/2: feature density + max-activating examples
# ---------------------------------------------------------------------------


@dataclass
class ResidueRecord:
    protein_id: str
    source: str
    position: int  # 0-indexed within the protein
    context: str  # local sequence window, center residue in [brackets]


def make_context(sequence: str, position: int, window: int) -> str:
    lo, hi = max(0, position - window), min(len(sequence), position + window + 1)
    return f"{sequence[lo:position]}[{sequence[position]}]{sequence[position + 1:hi]}"


def sample_residues(pool_df: pd.DataFrame, max_residues: int, seed: int) -> pd.DataFrame:
    """Stratified per-source random sample of residues (as global indices
    into activations.npy), so small sources (e.g. a few thousand natural-
    binder residues vs. ~2M design residues) aren't drowned out -- each
    source gets an equal share of the budget, capped at its own size."""
    rng = np.random.RandomState(seed)
    sources = sorted(pool_df["source"].unique())
    per_source_budget = max(1, max_residues // len(sources))

    picks = []
    for source in sources:
        rows = pool_df[pool_df["source"] == source]
        starts = rows["start"].to_numpy()
        lengths = rows["length"].to_numpy()
        ids = rows["id"].to_numpy()
        idx_parts = [np.arange(s, s + n) for s, n in zip(starts, lengths)]
        id_parts = [np.full(n, pid, dtype=object) for pid, n in zip(ids, lengths)]
        pos_parts = [np.arange(n) for n in lengths]
        all_idx = np.concatenate(idx_parts)
        all_id = np.concatenate(id_parts)
        all_pos = np.concatenate(pos_parts)

        n_take = min(per_source_budget, len(all_idx))
        chosen = rng.choice(len(all_idx), size=n_take, replace=False)
        picks.append(pd.DataFrame({
            "global_idx": all_idx[chosen],
            "id": all_id[chosen],
            "position": all_pos[chosen],
            "source": source,
        }))

    sampled = pd.concat(picks, ignore_index=True)
    sampled = sampled.merge(pool_df[["id", "sequence"]], on="id", how="left")
    return sampled.sort_values("global_idx").reset_index(drop=True)


class FeatureStats:
    """Streaming per-feature density + mean-activation-when-active. Avoids
    ever materializing the full (n_residues, d_hidden) code matrix."""

    def __init__(self, d_hidden: int):
        self.d_hidden = d_hidden
        self.fire_count = torch.zeros(d_hidden, dtype=torch.long)
        self.activation_sum = torch.zeros(d_hidden, dtype=torch.float64)
        self.n_seen = 0

    def update(self, codes: torch.Tensor) -> None:
        fired = codes != 0
        self.fire_count += fired.sum(dim=0).cpu()
        self.activation_sum += codes.sum(dim=0).double().cpu()
        self.n_seen += codes.shape[0]

    def to_dataframe(self) -> pd.DataFrame:
        fire = self.fire_count.numpy()
        density = fire / max(self.n_seen, 1)
        mean_active = np.divide(
            self.activation_sum.numpy(), fire, out=np.zeros(len(fire)), where=fire > 0
        )
        return pd.DataFrame({
            "feature": np.arange(self.d_hidden),
            "density": density,
            "fire_count": fire,
            "mean_activation_when_active": mean_active,
            "dead": fire == 0,
        })


class TopActivations:
    """Streaming top-N max-activating-example tracker: one small min-heap
    per feature, so we never hold more than d_hidden * top_n examples in
    memory regardless of pool size."""

    def __init__(self, d_hidden: int, top_n: int = 15):
        self.d_hidden = d_hidden
        self.top_n = top_n
        self._heaps: list[list[tuple[float, int, ResidueRecord]]] = [[] for _ in range(d_hidden)]
        self._tie = 0  # heapq tiebreaker -- ResidueRecord has no natural ordering

    def _offer(self, feature: int, activation: float, record: ResidueRecord) -> None:
        heap = self._heaps[feature]
        self._tie += 1
        entry = (activation, self._tie, record)
        if len(heap) < self.top_n:
            heapq.heappush(heap, entry)
        elif activation > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def update_batch(self, codes: torch.Tensor, records: list[ResidueRecord]) -> None:
        """codes: (B, d_hidden) sparse TopK activations, row-aligned with
        `records` (len B)."""
        nz = torch.nonzero(codes, as_tuple=False)
        if nz.numel() == 0:
            return
        values = codes[nz[:, 0], nz[:, 1]].tolist()
        for (row, feat), val in zip(nz.tolist(), values):
            self._offer(feat, val, records[row])

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for feat in range(self.d_hidden):
            ranked = sorted(self._heaps[feat], key=lambda e: -e[0])
            for rank, (activation, _, rec) in enumerate(ranked):
                rows.append({
                    "feature": feat,
                    "rank": rank,
                    "activation": activation,
                    "protein_id": rec.protein_id,
                    "source": rec.source,
                    "position": rec.position,
                    "context": rec.context,
                })
        return pd.DataFrame(rows)


def run_feature_interpretation(
    model, mean, scale, acts: np.memmap, pool_df: pd.DataFrame,
    max_residues: int, top_n: int, context: int, batch_size: int, seed: int, device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled = sample_residues(pool_df, max_residues, seed)
    print(f"Sampled {len(sampled)} residues across {sampled['source'].nunique()} sources for interpretation")

    stats = FeatureStats(model.d_hidden)
    top = TopActivations(model.d_hidden, top_n=top_n)

    for start in range(0, len(sampled), batch_size):
        chunk = sampled.iloc[start : start + batch_size]
        x = torch.from_numpy(acts[chunk["global_idx"].to_numpy()].astype(np.float32)).to(device)
        x_proc = center_scale(x, mean, scale)
        with torch.no_grad():
            codes = model.encode(x_proc)

        stats.update(codes)
        records = [
            ResidueRecord(
                protein_id=row.id,
                source=row.source,
                position=int(row.position),
                context=make_context(row.sequence, int(row.position), context),
            )
            for row in chunk.itertuples()
        ]
        top.update_batch(codes.cpu(), records)
        done = min(start + batch_size, len(sampled))
        if done % (batch_size * 10) == 0 or done == len(sampled):
            print(f"  {done}/{len(sampled)} residues encoded")

    return stats.to_dataframe(), top.to_dataframe()


# ---------------------------------------------------------------------------
# Step 3: linear probe (pooled per-protein codes -> binding metrics)
# ---------------------------------------------------------------------------


def encode_pool_per_protein(
    pool_df: pd.DataFrame, protein_ids, acts: np.memmap, model, mean, scale, device, batch_size: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Encodes EVERY residue of each requested protein (no subsampling --
    the probe needs each protein's complete pooled footprint). Returns
    (ids, max_pooled, mean_pooled), each pooled array (n_proteins, d_hidden).

    Both poolings are kept: max-pool answers "does this feature fire
    ANYWHERE in this design" (closer to a motif detector); mean-pool
    answers "how much of this design's surface does this feature cover"
    (closer to a bulk property). Either can end up more predictive."""
    protein_ids = set(protein_ids)
    rows = pool_df[pool_df["id"].isin(protein_ids)].sort_values("start")

    ids, max_pooled, mean_pooled = [], [], []
    for row in rows.itertuples():
        idx = np.arange(row.start, row.start + row.length)
        code_parts = []
        for s in range(0, len(idx), batch_size):
            chunk_idx = idx[s : s + batch_size]
            x = torch.from_numpy(acts[chunk_idx].astype(np.float32)).to(device)
            x_proc = center_scale(x, mean, scale)
            with torch.no_grad():
                code_parts.append(model.encode(x_proc).cpu())
        codes = torch.cat(code_parts, dim=0)
        ids.append(row.id)
        max_pooled.append(codes.max(dim=0).values.numpy())
        mean_pooled.append(codes.mean(dim=0).numpy())

    return ids, np.stack(max_pooled), np.stack(mean_pooled)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values. Needed because the univariate scan runs
    d_hidden (thousands of) independent correlation tests -- at nominal
    p<0.05 with NO features actually predictive, ~5% of them (hundreds)
    would still cross that threshold by chance alone."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def run_linear_probe(
    ids: list, pooled_codes: np.ndarray, metrics_df: pd.DataFrame, target_col: str, cv: int = 5, seed: int = 0,
) -> dict:
    """Two complementary views of feature-vs-metric relationship:

      1. Univariate: Spearman rank correlation of each feature against the
         target, independently. Cheap, robust to outliers, easy to sanity
         check by eye -- but blind to features that only matter jointly.
      2. Multivariate: Lasso (L1-regularized) linear regression predicting
         the target from ALL pooled features at once, with nested cross-
         validation for an honest R^2 (LassoCV's own inner CV picks alpha
         inside each outer fold, so the reported R^2 is a genuine held-out
         score, not inflated by alpha-selection leakage). L1 specifically
         because d_hidden (thousands) vastly exceeds n_proteins (typically
         a few hundred held-out designs) -- plain OLS is underdetermined;
         L1 drives most coefficients to exactly zero, leaving a short list
         of candidate binding-relevant feature ids to go look up in
         feature_top_examples.csv.

    Requires scikit-learn + scipy (not otherwise dependencies of this
    package -- only needed if you pass --probe-metrics-csv).
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score

    y_series = metrics_df.set_index("id")[target_col].reindex(ids)
    valid = y_series.notna().to_numpy()
    if valid.sum() < 10:
        raise ValueError(
            f"Only {int(valid.sum())} of {len(ids)} probed proteins have a non-null "
            f"'{target_col}' -- too few to probe (need >=10)."
        )
    X = pooled_codes[valid]
    y = y_series.to_numpy()[valid].astype(float)

    # 1. Univariate.
    rhos, pvals = [], []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(col == col[0]):  # constant column (e.g. a globally-dead feature) -- corr undefined
            rhos.append(0.0)
            pvals.append(1.0)
            continue
        rho, p = spearmanr(col, y)
        rhos.append(0.0 if np.isnan(rho) else rho)
        pvals.append(1.0 if np.isnan(p) else p)
    univariate = pd.DataFrame({
        "feature": np.arange(X.shape[1]),
        "spearman_rho": rhos,
        "p_value": pvals,
        "q_value": _benjamini_hochberg(np.array(pvals)),
    })
    univariate = univariate.reindex(univariate["spearman_rho"].abs().sort_values(ascending=False).index)

    # 2. Multivariate, nested CV.
    n_splits = min(cv, len(y))
    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probe_model = LassoCV(cv=inner_kf, random_state=seed, max_iter=10_000, n_jobs=-1)
    oof_pred = cross_val_predict(probe_model, X, y, cv=outer_kf)
    honest_r2 = r2_score(y, oof_pred)

    final_model = LassoCV(cv=inner_kf, random_state=seed, max_iter=10_000, n_jobs=-1).fit(X, y)
    nonzero = np.nonzero(final_model.coef_)[0]
    multivariate = pd.DataFrame({"feature": nonzero, "lasso_coef": final_model.coef_[nonzero]})
    multivariate = multivariate.reindex(multivariate["lasso_coef"].abs().sort_values(ascending=False).index)

    return {
        "target": target_col,
        "n_proteins": int(valid.sum()),
        "cv_folds": n_splits,
        "cross_validated_r2": float(honest_r2),
        "lasso_alpha": float(final_model.alpha_),
        "univariate": univariate,
        "multivariate": multivariate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-residues", type=int, default=200_000)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--context", type=int, default=6, help="Residues of sequence shown on each side of a max-activating residue.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-metrics-csv", type=Path, default=None, help="CSV with an 'id' column plus binding-metric columns (e.g. from rank_boltz_results.py + compute_ipsae.py/compute_ipae.py). Omit to skip the linear probe.")
    parser.add_argument("--probe-targets", type=str, default="binding_confidence,iptm,ipsae,ipae")
    parser.add_argument(
        "--probe-protein-set", choices=["val", "all"], default="val",
        help="'val' (default): the checkpoint's held-out validation proteins, never seen "
        "during training -- the defensible choice for testing whether features generalize. "
        "'all': every design protein in the pool (train+val), for more probe power at the "
        "cost of testing on proteins the SAE was fit on.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, mean, scale, val_protein_ids = load_sae(args.checkpoint, device)
    print(f"Loaded SAE: d_model={model.d_model} d_hidden={model.d_hidden} k={model.k}")

    pool_df = load_pool(args.data_dir)
    acts = np.load(args.data_dir / "activations.npy", mmap_mode="r")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Feature interpretation: density + max-activating examples ===")
    stats_df, top_df = run_feature_interpretation(
        model, mean, scale, acts, pool_df,
        args.max_residues, args.top_n, args.context, args.batch_size, args.seed, device,
    )
    stats_df.to_csv(args.output_dir / "feature_stats.csv", index=False)
    top_df.to_csv(args.output_dir / "feature_top_examples.csv", index=False)
    n_dead = int(stats_df["dead"].sum())
    print(f"Wrote feature_stats.csv ({n_dead}/{model.d_hidden} dead) and feature_top_examples.csv")

    if args.probe_metrics_csv is None:
        print("\nNo --probe-metrics-csv given, skipping linear probe.")
        return

    print("\n=== Linear probe: pooled per-protein codes vs. binding metrics ===")
    metrics_df = pd.read_csv(args.probe_metrics_csv)
    assert "id" in metrics_df.columns, f"{args.probe_metrics_csv} must have an 'id' column"

    if args.probe_protein_set == "val":
        probe_ids = val_protein_ids.tolist()
    else:
        probe_ids = pool_df.loc[~pool_df["source"].isin(EVAL_ONLY_SOURCES), "id"].unique().tolist()
    print(f"Probing {len(probe_ids)} proteins ({args.probe_protein_set} set)")

    ids, max_pooled, mean_pooled = encode_pool_per_protein(
        pool_df, probe_ids, acts, model, mean, scale, device, args.batch_size
    )

    for target in args.probe_targets.split(","):
        target = target.strip()
        if target not in metrics_df.columns:
            print(f"  '{target}' not in {args.probe_metrics_csv}, skipping")
            continue
        for pool_name, pooled in [("max", max_pooled), ("mean", mean_pooled)]:
            try:
                result = run_linear_probe(ids, pooled, metrics_df, target, seed=args.seed)
            except ValueError as e:
                print(f"  {target} ({pool_name}-pool): {e}")
                continue
            print(
                f"  {target} ({pool_name}-pool): n={result['n_proteins']} "
                f"cv_r2={result['cross_validated_r2']:.4f} "
                f"nonzero_lasso_features={len(result['multivariate'])}"
            )
            result["univariate"].to_csv(args.output_dir / f"probe_{target}_{pool_name}_univariate.csv", index=False)
            result["multivariate"].to_csv(args.output_dir / f"probe_{target}_{pool_name}_multivariate.csv", index=False)
            summary = {k: v for k, v in result.items() if k not in ("univariate", "multivariate")}
            with open(args.output_dir / f"probe_{target}_{pool_name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
