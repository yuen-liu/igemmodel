"""LLM auto-labeling of SAE features from their max-activating examples.

Downstream of feature_analysis.py, not a replacement for it: this script
reads feature_top_examples.csv (the per-feature top-N max-activating
residues, with local sequence context) and asks an LLM to draft a one-
sentence description of each feature's pattern, purely from that evidence.

KNOWN LIMITATION, deliberately deferred: this only sends local SEQUENCE
context (+/- a few residues), not structural context (secondary structure,
solvent burial, distance to the binder-target interface). A feature whose
real basis is structural (e.g. "fires on buried core residues") may look
like pure noise in a sequence-only window, or worse, an interpreter can
latch onto a spurious sequence coincidence that isn't the real reason it
fires. Boltz's predicted CIFs already carry per-residue pLDDT (B-factor
column) and 3D coordinates that could add cheap structural context (no new
dependency -- same CIF-parsing pattern already in
notebooks/bridget/esmc_embedding_analysis/esmc_embedding_analysis_vilip1.ipynb's
structural-sanity-check section), but the results/ directory layout is
per-campaign and not available on this machine (data/ is gitignored) --
revisit after seeing whether sequence-only labels are useful at all.

This is a scale tool, not a validation tool -- treat every label as an
unverified hypothesis a human should sanity-check (e.g. against structure),
not a finding. The LLM sees exactly what a human reading
feature_top_examples.csv would see: each feature's own precomputed top-N
rows (protein id, source, position, activation, local sequence window) --
nothing else. No cross-feature context, no full protein sequences, no 3D
structure -- see feature_analysis.py's module docstring for why only local
sequence context is captured at all.

Uses the Message Batches API (not per-feature synchronous calls): this is a
bulk, non-interactive, non-latency-sensitive job -- the textbook case for
batching, and it's 50% cheaper. A batch can take up to a few hours to
complete; this script polls until done.

Requires `pip install anthropic` and an API key (ANTHROPIC_API_KEY env var,
or `ant auth login`) -- NOT covered by a Claude.ai Pro/Max subscription,
which only covers claude.ai/Claude Code usage, not programmatic API calls.

Usage:
    # Pilot on a handful of features first (near-zero cost, read the output
    # yourself before trusting it):
    python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --output-dir feature_analysis_results --features 12,845,2001

    # Scope to the linear probe's candidate features (recommended default use):
    python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --output-dir feature_analysis_results \\
        --candidates-csv feature_analysis_results/probe_ipsae_max_multivariate.csv

    # Every feature that has at least one recorded example (full dictionary,
    # ~$3 on Haiku 4.5 with batching at dict_size=4096 -- see module docstring):
    python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --output-dir feature_analysis_results

    # Preview the exact prompt for the first feature without spending anything:
    python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --output-dir feature_analysis_results --features 12 --dry-run

    # With InterPro annotations layered in (see fetch_interpro.py) -- used as
    # supplementary evidence when present, sequence-context-only fallback
    # when not (expected for most examples -- see fetch_interpro.py's
    # module docstring on why de novo designs mostly won't have hits):
    python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --output-dir feature_analysis_results --features 12,845,2001 \\
        --interpro-csv feature_analysis_results/interpro_annotations.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are analyzing features from a sparse autoencoder (SAE) trained on ESM-C \
protein language model activations. Each feature is a learned direction that fires on certain \
residues. You will see the residues that activate ONE specific feature most strongly, as short \
local sequence windows with the activating residue in [brackets] -- e.g. "ABCD[E]FGHI" means \
residue E fired the feature, with D/C/B/A before it and F/G/H/I after.

You may also see an "InterPro evidence" section: a tally of real database domain/family \
annotations, one per example, each looked up at that residue's exact position in its full \
protein. Most proteins in this dataset are computationally designed and have NO evolutionary \
relationship to any characterized protein family, so most or all examples showing "No InterPro \
annotation" is expected and NOT evidence of anything -- do not comment on its absence. If one \
InterPro annotation clearly dominates the tally (covers a majority of the annotated examples), \
treat it as strong, verified evidence and prefer describing the pattern in those terms over \
guessing from sequence alone. A minority or scattered set of different annotations is weak \
evidence -- weigh it accordingly, do not report the single most common one as if it were \
consensus when it only covers a few examples.

Describe in ONE sentence the shared pattern across these examples, if any: a dominant InterPro \
domain match, an amino acid identity, a local motif (e.g. adjacent to a specific residue or a \
small recurring subsequence), or a position tendency. Base your answer strictly on the evidence \
given -- do not speculate about biological function, structure, or binding beyond what the \
evidence shows. If the examples show no consistent pattern, say exactly: "No clear pattern." Do \
not add any other text."""


def aggregate_interpro(rows: pd.DataFrame) -> str | None:
    """Deterministic tally of InterPro annotations across a feature's
    examples, computed in code rather than left for the LLM to count from
    scattered per-row text -- the tally itself should never be wrong."""
    if "interpro_annotation" not in rows.columns:
        return None
    n = len(rows)
    counts = rows["interpro_annotation"].value_counts(dropna=True)
    n_annotated = int(counts.sum())
    if n_annotated == 0:
        return None
    lines = [f"InterPro evidence across the {n} examples (each looked up at that residue's exact position in its full protein):"]
    for annotation, count in counts.items():
        lines.append(f"- {annotation}: {count}/{n} examples")
    n_missing = n - n_annotated
    if n_missing:
        lines.append(f"- No InterPro annotation: {n_missing}/{n} examples")
    return "\n".join(lines)


def build_user_prompt(feature_id: int, examples: pd.DataFrame, top_n: int) -> str:
    rows = examples[examples["feature"] == feature_id].sort_values("rank").head(top_n)
    lines = [
        f"Feature {feature_id}, top {len(rows)} activating residues (sequence context):",
    ]
    for i, row in enumerate(rows.itertuples(), start=1):
        lines.append(f"{i}. activation={row.activation:.3f} source={row.source} context={row.context}")

    interpro_summary = aggregate_interpro(rows)
    if interpro_summary:
        lines.append("")
        lines.append(interpro_summary)

    return "\n".join(lines)


def load_candidate_features(args, examples: pd.DataFrame) -> list[int]:
    if args.features:
        return [int(f.strip()) for f in args.features.split(",")]
    if args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        return sorted(candidates_df["feature"].unique().tolist())
    return sorted(examples["feature"].unique().tolist())


def submit_batch(client, requests: list) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch_requests = [
        Request(
            custom_id=f"feature-{feature_id}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ),
        )
        for feature_id, user_prompt, model, max_tokens in requests
    ]
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Submitted batch {batch.id} ({len(batch_requests)} requests), status={batch.processing_status}")
    return batch.id


def poll_batch(client, batch_id: str, poll_seconds: int) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            print(f"Batch {batch_id} ended. Counts: {batch.request_counts}")
            return
        print(
            f"  {batch.processing_status}: "
            f"processing={batch.request_counts.processing} "
            f"succeeded={batch.request_counts.succeeded} "
            f"errored={batch.request_counts.errored}"
        )
        time.sleep(poll_seconds)


def collect_results(client, batch_id: str) -> pd.DataFrame:
    rows = []
    for result in client.messages.batches.results(batch_id):
        feature_id = int(result.custom_id.removeprefix("feature-"))
        if result.result.type == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"), ""
            ).strip()
            rows.append({"feature": feature_id, "label": text, "error": None})
        else:
            # errored / canceled / expired -- see python/claude-api/batches.md's
            # result-handling pattern for what each of these means.
            rows.append({"feature": feature_id, "label": None, "error": result.result.type})
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-csv", type=Path, required=True, help="feature_analysis.py's feature_top_examples.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature ids to label (pilot mode). Overrides --candidates-csv.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="A probe_*_multivariate.csv or probe_*_univariate.csv -- labels exactly the features listed in its 'feature' column.")
    parser.add_argument("--top-n", type=int, default=15, help="How many of each feature's saved examples to include in its prompt (<= feature_analysis.py's --top-n).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--interpro-csv", type=Path, default=None, help="fetch_interpro.py's interpro_annotations.csv -- adds InterPro domain/family evidence per example where available (expected to be sparse; see fetch_interpro.py's docstring).")
    parser.add_argument("--dry-run", action="store_true", help="Print request count and the first feature's exact prompt, then exit -- no API call, no cost.")
    args = parser.parse_args()

    examples = pd.read_csv(args.examples_csv)
    if args.interpro_csv:
        interpro = pd.read_csv(args.interpro_csv)
        examples = examples.merge(
            interpro[["feature", "rank", "protein_id", "position", "interpro_annotation"]],
            on=["feature", "rank", "protein_id", "position"],
            how="left",
        )
        n_hits = examples["interpro_annotation"].notna().sum()
        print(f"Merged InterPro annotations: {n_hits}/{len(examples)} example rows have a hit")

    feature_ids = load_candidate_features(args, examples)
    print(f"Labeling {len(feature_ids)} feature(s) from {args.examples_csv}")

    requests = [
        (fid, build_user_prompt(fid, examples, args.top_n), args.model, args.max_tokens)
        for fid in feature_ids
    ]

    if args.dry_run:
        print(f"\nWould submit {len(requests)} requests to {args.model}. Example prompt (feature {requests[0][0]}):\n")
        print("--- system ---")
        print(SYSTEM_PROMPT)
        print("\n--- user ---")
        print(requests[0][1])
        return

    import anthropic

    client = anthropic.Anthropic()
    batch_id = submit_batch(client, requests)
    poll_batch(client, batch_id, args.poll_seconds)

    results = collect_results(client, batch_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "feature_labels.csv"
    results.to_csv(out_path, index=False)

    n_ok = results["error"].isna().sum()
    print(f"\n{n_ok}/{len(results)} labeled successfully. Wrote {out_path}")


if __name__ == "__main__":
    main()
