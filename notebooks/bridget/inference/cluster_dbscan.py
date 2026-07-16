"""DBSCAN clustering over ESM-C embeddings, with top-scoring representative
selection per cluster.

Generic over pool size -- works unchanged on the top-500 embeddings or the
full 20k once available, as long as the embeddings.npz/manifest.csv pair
line up by `id`.

Embeddings are L2-normalized and clustered with Euclidean distance, which is
equivalent to cosine distance on normalized vectors but lets sklearn use its
faster ball-tree backend instead of falling back to brute force.

Usage:
    python cluster_dbscan.py --embeddings embeddings.npz --manifest manifest.csv \
        --output clusters.csv --eps 0.3 --min-samples 5

    # Also pick the best `top-k-per-cluster` designs (by binding_confidence)
    # from each cluster, for the diversity-subsampling step:
    python cluster_dbscan.py --embeddings embeddings.npz --manifest manifest.csv \
        --output clusters.csv --selected-output selected.csv --top-k-per-cluster 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=None,
        help="If set, also write top-k-per-cluster representatives here.",
    )
    parser.add_argument("--top-k-per-cluster", type=int, default=1)
    parser.add_argument(
        "--rank-by",
        default="binding_confidence",
        help="Manifest column to rank representatives within each cluster (higher=better).",
    )
    args = parser.parse_args()

    npz = np.load(args.embeddings, allow_pickle=True)
    ids = npz["ids"]
    embeddings = npz["embeddings"]
    print(f"Loaded {len(ids)} embeddings, dim={embeddings.shape[1]}")

    manifest = pd.read_csv(args.manifest)
    emb_df = pd.DataFrame({"id": ids})
    emb_df["_emb_idx"] = np.arange(len(ids))
    merged = emb_df.merge(manifest, on="id", how="left")

    n_missing = merged[args.rank_by].isna().sum() if args.rank_by in merged else len(merged)
    if n_missing:
        print(
            f"Warning: {n_missing}/{len(merged)} rows missing '{args.rank_by}' "
            "from the manifest (id mismatch or column absent)"
        )

    normalized = normalize(embeddings, norm="l2")
    # eps here is a Euclidean distance threshold; since vectors are unit-norm,
    # euclidean_dist^2 = 2 * (1 - cosine_similarity), so eps=0.3 roughly
    # corresponds to requiring cosine similarity above ~0.955. Tune eps based
    # on the diagnostics printed below, not blindly.
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="euclidean")
    labels = clustering.fit_predict(normalized)
    merged["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points (cluster=-1): {n_noise}/{len(labels)} ({n_noise / len(labels):.1%})")
    if n_clusters:
        sizes = pd.Series(labels[labels != -1]).value_counts()
        print(f"Cluster size distribution: min={sizes.min()} median={sizes.median():.0f} max={sizes.max()}")

    merged.drop(columns=["_emb_idx"]).to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows with cluster assignments to {args.output}")

    if args.selected_output is not None:
        if args.rank_by not in merged:
            raise ValueError(
                f"Cannot select representatives: '{args.rank_by}' not found in manifest columns "
                f"({list(manifest.columns)})"
            )
        clustered = merged[merged["cluster"] != -1]
        selected = (
            clustered.sort_values(args.rank_by, ascending=False)
            .groupby("cluster")
            .head(args.top_k_per_cluster)
            .sort_values(["cluster", args.rank_by], ascending=[True, False])
        )
        selected.drop(columns=["_emb_idx"], errors="ignore").to_csv(
            args.selected_output, index=False
        )
        print(
            f"Wrote {len(selected)} representatives "
            f"(top {args.top_k_per_cluster} per cluster, ranked by {args.rank_by}) "
            f"to {args.selected_output}"
        )


if __name__ == "__main__":
    main()
