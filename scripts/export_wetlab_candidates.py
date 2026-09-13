"""Export wet-lab candidate CSVs + structure files from the picks already
made in the esmc_deepdive notebooks (deep-dive layer chosen by each
target's esmc_layer_sweep_*.ipynb):

  fabp7 -> esmc_deepdive_fabp7_layer23.ipynb
  uchl1 -> esmc_deepdive_uchl1_layer19.ipynb
  vilip1 -> esmc_deepdive_vilip1_composite_hotspot_combined_layer18.ipynb
            (combined 45k campaign, layer 18 -- supersedes the v2-only
            layer20 notebook and the original full20k layer26 notebook)

Those notebooks already ran and saved two selections per target under
data/<campaign>/clustering/layer_<N>/:
  dunbrack_top5_per_cluster.csv      -- best 5 per HDBSCAN cluster by
                                         ipsae/iptm/ipae (missing `sequence`
                                         and a few manifest columns)
  naive_top5_binding_confidence.csv  -- naive top 5 by binding_confidence
                                         across the whole unfiltered pool

This script does NOT recompute any ranking -- it just reads those two CSVs,
dedupes, fills in `sequence` (and other manifest columns) for the dunbrack
rows via a manifest join, and pulls each pick's structure file into a small
per-target bundle so wet lab doesn't have to dig through the 11-40k-design
result trees (several of which live on Google Drive, not locally) to find
one candidate's .cif by id.

Output: data/wetlab_export/<target>_candidates.csv (+ structures/<id>.cif)
and data/wetlab_export/all_targets_candidates.csv combined.
data/ is already gitignored, so this isn't tracked or committed.
"""

import os
import shutil
import tarfile
import tempfile

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "data", "wetlab_export")

# Google Drive Desktop mount for the vilip1 "new 40k" (20260728) campaign.
# NOTE: the repo-root symlink `vilip1-20260728` is stale -- Drive renamed
# the shared folder to `vilip1-20260728-v1` after the symlink was created,
# so it resolves to a folder that no longer exists. Pointing at the actual
# path directly instead. If Drive re-syncs this under a different path,
# update GD_VILIP1_40K below.
GD_VILIP1_40K = (
    "/Users/bridget/Library/CloudStorage/GoogleDrive-bgl2126@columbia.edu"
    "/.shortcut-targets-by-id/1JJajJWDRB-uFFztgbLdJHJnK4l29TzP1"
    "/vilip1-20260728-v1/results"
)


def cif_from_files_breakout(results_dir, design_id, cif_name):
    """Standard layout: <results_dir>/<id>/files/result/<cif_name>."""
    path = os.path.join(results_dir, design_id, "files", "result", cif_name)
    return path if os.path.exists(path) else None


def cif_from_archive(results_dir, design_id, dest_path):
    """Fallback for the 20260728 tree: the files/result/ breakout didn't
    fully sync down from Drive for most designs (~120k tiny files), but
    archive.tar.gz per design did. Extract predicted_structure.cif from it
    straight to dest_path."""
    archive_path = os.path.join(results_dir, design_id, "archive.tar.gz")
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
        return False
    try:
        with tarfile.open(archive_path, "r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
            tar.extractall(tmp, filter="data")
            extracted = os.path.join(tmp, "result", "predicted_structure.cif")
            if not os.path.exists(extracted):
                return False
            shutil.copyfile(extracted, dest_path)
            return True
    except (tarfile.TarError, OSError):
        return False


def make_simple_resolver(results_dir):
    """Resolver for the two local-only campaigns (fabp7, uchl1): single
    results tree, standard files/result/<id>_predicted.cif layout."""

    def resolve(design_id, dest_path):
        src = cif_from_files_breakout(results_dir, design_id, f"{design_id}_predicted.cif")
        if not src:
            return False
        shutil.copyfile(src, dest_path)
        return True

    return resolve


def resolve_vilip1_structure(design_id, dest_path):
    v2_dir = os.path.join(REPO_ROOT, "data", "vilip1-design-composite_hotspot-20260724-v2", "results")
    src = cif_from_files_breakout(v2_dir, design_id, f"{design_id}_predicted.cif")
    if src:
        shutil.copyfile(src, dest_path)
        return True

    src = cif_from_files_breakout(GD_VILIP1_40K, design_id, f"{design_id}_predicted.cif")
    if src:
        shutil.copyfile(src, dest_path)
        return True

    return cif_from_archive(GD_VILIP1_40K, design_id, dest_path)


TARGETS = [
    {
        "name": "fabp7",
        "selection_dir": "data/fabp7_full20k/clustering/layer_23",
        "manifest": "data/fabp7_full20k/manifest_with_ipsae.csv",
        "resolve_structure": make_simple_resolver(os.path.join(REPO_ROOT, "data/fabp7_full20k/results")),
    },
    {
        "name": "uchl1",
        "selection_dir": "data/UCH_L1_final/clustering/layer_19",
        "manifest": "data/UCH_L1_final/manifest_with_ipsae.csv",
        "resolve_structure": make_simple_resolver(os.path.join(REPO_ROOT, "data/UCH_L1_final/results")),
    },
    {
        "name": "vilip1",
        "selection_dir": "data/vilip1-design-composite_hotspot-combined/clustering/layer_18",
        "manifest": "data/vilip1-design-composite_hotspot-combined/manifest_with_ipsae.csv",
        "resolve_structure": resolve_vilip1_structure,
    },
]

MANIFEST_JOIN_COLUMNS = ["sequence", "min_interaction_pae", "structure_confidence", "complex_plddt"]
CSV_COLUMNS = [
    "selection",
    "cluster",
    "id",
    "sequence",
    "binding_confidence",
    "iptm",
    "ipsae",
    "ipae",
    "min_interaction_pae",
    "structure_confidence",
    "complex_plddt",
]


def export_target(target):
    selection_dir = os.path.join(REPO_ROOT, target["selection_dir"])
    manifest = pd.read_csv(os.path.join(REPO_ROOT, target["manifest"]))

    dunbrack = pd.read_csv(os.path.join(selection_dir, "dunbrack_top5_per_cluster.csv"))
    dunbrack["selection"] = "dunbrack_top5_per_cluster"
    dunbrack = dunbrack.merge(manifest[["id"] + MANIFEST_JOIN_COLUMNS], on="id", how="left")

    naive = pd.read_csv(os.path.join(selection_dir, "naive_top5_binding_confidence.csv"))
    naive["selection"] = "naive_top5_binding_confidence"
    naive["cluster"] = None

    combined = pd.concat([dunbrack, naive], ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="first")

    target_dir = os.path.join(OUTPUT_ROOT, target["name"])
    structures_dir = os.path.join(target_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    csv_path = os.path.join(OUTPUT_ROOT, f"{target['name']}_candidates.csv")
    combined[CSV_COLUMNS].to_csv(csv_path, index=False)
    print(f"[{target['name']}] {len(combined)} unique candidates -> {csv_path}")

    missing = []
    for design_id in combined["id"]:
        dest = os.path.join(structures_dir, f"{design_id}.cif")
        if not target["resolve_structure"](design_id, dest):
            missing.append(design_id)

    if missing:
        print(f"[{target['name']}] WARNING: could not resolve structure files for {len(missing)} ids:")
        for design_id in missing:
            print(f"  - {design_id}")
    print(f"[{target['name']}] copied {len(combined) - len(missing)}/{len(combined)} structure files to {structures_dir}")

    combined = combined.copy()
    combined.insert(0, "target", target["name"])
    return combined[["target"] + CSV_COLUMNS]


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_combined = [export_target(target) for target in TARGETS]
    combined = pd.concat(all_combined, ignore_index=True)
    combined_path = os.path.join(OUTPUT_ROOT, "all_targets_candidates.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nWrote combined manifest: {combined_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
