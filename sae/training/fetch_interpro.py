"""Fetch InterPro domain/family/site annotations for the proteins behind a
feature's max-activating examples, via EBI's InterProScan5 REST API, and
cross-reference them against the SPECIFIC residue position that fired the
feature (not just "does this protein have any annotation at all").

Coverage caveat, expected and not a bug: this corpus is overwhelmingly de
novo Boltz-designed binders (~25k), not evolved natural proteins (13-82,
the natural_binders/binder_dataset_vilip1 sources). InterPro's member
databases (Pfam, PROSITE, SMART, ...) are profile HMMs built from
evolutionary conservation across natural homologs -- a de novo design has
no evolutionary relationship to any characterized family, even when it
successfully mimics a natural fold, so most design-source examples are
expected to come back with NO hit. Coverage should concentrate on the
natural-binder examples. label_features.py is meant to use this as
supplementary evidence when available and fall back to sequence-context-only
reasoning when it isn't -- not to require an InterPro hit to produce a label.

REST API shape verified directly against EBI's live service in this session
(not from documentation alone -- the docs pages don't expose the JSON
schema): POST {BASE_URL}/run (form fields email, sequence, title) returns a
bare job id string; GET {BASE_URL}/status/{job_id} returns
QUEUED/RUNNING/FINISHED/ERROR/FAILURE/NOT_FOUND; GET
{BASE_URL}/result/{job_id}/json returns {"results": [{"matches": [...]}]}
where each match has signature.entry.accession/name (the actual InterPro
accession -- signature.accession is the member-database id, e.g. a Pfam
id, not the InterPro one) and locations: [{"start", "end", ...}], 1-indexed
inclusive (confirmed against a real completed job: a PF03590/IPR004618 hit
at start=8, end=326 on a real protein).

Fair-use note: this is a shared, free EBI research service with no true
batch endpoint -- one job per unique protein sequence, each taking minutes.
Sized for a few dozen candidate proteins (e.g. the linear probe's
feature list), not the full ~25k-protein corpus; dedupes by protein id so
proteins shared across multiple features' example lists are only submitted
once.

Usage:
    python fetch_interpro.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --manifest manifest_combined.csv --output-dir feature_analysis_results \\
        --email you@example.com --features 12,845,2001

    # scope to the probe's candidate features instead of a manual list:
    python fetch_interpro.py --examples-csv feature_analysis_results/feature_top_examples.csv \\
        --manifest manifest_combined.csv --output-dir feature_analysis_results \\
        --email you@example.com --candidates-csv feature_analysis_results/probe_ipsae_max_multivariate.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


def submit_job(sequence: str, email: str, title: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/run",
        data={"email": email, "sequence": sequence, "title": title, "goterms": "false", "pathways": "false"},
    )
    resp.raise_for_status()
    return resp.text.strip()


def poll_jobs(jobs: dict, poll_seconds: int, max_polls: int) -> dict:
    """jobs: {protein_id: job_id}. Returns {protein_id: final_status}, where
    status is FINISHED / ERROR / FAILURE / NOT_FOUND / TIMEOUT."""
    pending = dict(jobs)
    statuses = {}
    for poll_i in range(max_polls):
        if not pending:
            break
        for protein_id, job_id in list(pending.items()):
            resp = requests.get(f"{BASE_URL}/status/{job_id}")
            status = resp.text.strip()
            if status in ("FINISHED", "ERROR", "FAILURE", "NOT_FOUND"):
                statuses[protein_id] = status
                del pending[protein_id]
        if pending:
            print(f"  poll {poll_i + 1}: {len(pending)}/{len(jobs)} still running")
            time.sleep(poll_seconds)
    for protein_id in pending:
        statuses[protein_id] = "TIMEOUT"
    return statuses


def fetch_matches(job_id: str) -> list:
    resp = requests.get(f"{BASE_URL}/result/{job_id}/json")
    resp.raise_for_status()
    return resp.json()["results"][0]["matches"]


def annotate_position(matches: list, position_0indexed: int) -> str | None:
    """InterPro locations are 1-indexed inclusive; feature_top_examples.csv's
    `position` column is 0-indexed (data.py's _expand_to_residue_indices) --
    convert before comparing. Returns the first covering InterPro entry's
    "name (accession)", or None if no match covers this position (either no
    hits on the protein at all, or hits elsewhere in the protein but not at
    this specific residue)."""
    pos_1indexed = position_0indexed + 1
    for match in matches:
        entry = match["signature"].get("entry")
        if entry is None:  # signature not yet integrated into an InterPro entry
            continue
        for loc in match["locations"]:
            if loc["start"] <= pos_1indexed <= loc["end"]:
                return f"{entry['name']} ({entry['accession']})"
    return None


def load_candidate_features(args) -> list:
    if args.features:
        return [int(f.strip()) for f in args.features.split(",")]
    if args.candidates_csv:
        candidates_df = pd.read_csv(args.candidates_csv)
        assert "feature" in candidates_df.columns, f"{args.candidates_csv} has no 'feature' column"
        return sorted(candidates_df["feature"].unique().tolist())
    return None  # caller: no filter, use every feature in examples-csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples-csv", type=Path, required=True, help="feature_analysis.py's feature_top_examples.csv")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest_combined.csv (id, sequence, source) -- for full sequences")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--email", type=str, required=True, help="Required by EBI's usage policy -- use your own contact email, not a shared/placeholder one.")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature ids. Overrides --candidates-csv.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="A probe_*_multivariate.csv or probe_*_univariate.csv -- scope to exactly its 'feature' column.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-polls", type=int, default=40, help="Give up waiting on a job after this many polls (default: ~20 min at --poll-seconds 30).")
    parser.add_argument("--submit-delay-seconds", type=float, default=3.0, help="Pause between job submissions -- be a polite citizen of a shared free service.")
    args = parser.parse_args()

    examples = pd.read_csv(args.examples_csv)
    feature_ids = load_candidate_features(args)
    if feature_ids is not None:
        examples = examples[examples["feature"].isin(feature_ids)]
        print(f"Scoped to {len(feature_ids)} feature(s), {len(examples)} example rows")
    else:
        print(f"No feature filter given -- using all {examples['feature'].nunique()} features, {len(examples)} example rows")

    manifest = pd.read_csv(args.manifest)[["id", "sequence"]]
    unique_protein_ids = examples["protein_id"].unique().tolist()
    proteins = manifest[manifest["id"].isin(unique_protein_ids)]
    missing = set(unique_protein_ids) - set(proteins["id"])
    if missing:
        print(f"WARNING: {len(missing)} protein id(s) in examples-csv not found in manifest -- skipping: {sorted(missing)[:5]}...")

    print(f"Submitting {len(proteins)} unique-protein InterProScan job(s) to EBI...")
    jobs = {}
    for row in proteins.itertuples():
        job_id = submit_job(row.sequence, args.email, title=f"sae_feature_{row.id}"[:100])
        jobs[row.id] = job_id
        time.sleep(args.submit_delay_seconds)
    print(f"All {len(jobs)} jobs submitted. Polling for completion (this can take several minutes)...")

    statuses = poll_jobs(jobs, args.poll_seconds, args.max_polls)
    n_finished = sum(1 for s in statuses.values() if s == "FINISHED")
    print(f"{n_finished}/{len(statuses)} jobs finished. Others: "
          f"{ {s: sum(1 for v in statuses.values() if v == s) for s in set(statuses.values()) if s != 'FINISHED'} }")

    matches_by_protein = {}
    for protein_id, status in statuses.items():
        if status != "FINISHED":
            continue
        try:
            matches_by_protein[protein_id] = fetch_matches(jobs[protein_id])
        except Exception as e:
            print(f"  failed to fetch results for {protein_id}: {e}")

    rows = []
    for row in examples.itertuples():
        matches = matches_by_protein.get(row.protein_id)
        annotation = annotate_position(matches, row.position) if matches is not None else None
        rows.append({
            "feature": row.feature,
            "rank": row.rank,
            "protein_id": row.protein_id,
            "position": row.position,
            "interpro_annotation": annotation,
        })
    out_df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "interpro_annotations.csv"
    out_df.to_csv(out_path, index=False)
    n_hits = out_df["interpro_annotation"].notna().sum()
    print(f"\n{n_hits}/{len(out_df)} example residues fell inside an InterPro-annotated region. Wrote {out_path}")


if __name__ == "__main__":
    main()
