# SENTINEL: interpretable SAEs for protein binder design

Stroke remains a leading cause of disability and death worldwide, with
therapeutic efficacy declining rapidly post-onset. Circulating biomarkers
offer a path to early diagnosis, but designing high-affinity biosensing
binders is constrained by the opacity of generative protein models -- they
operate as black boxes, obscuring the features that actually drive high
affinity and specificity.

SENTINEL addresses this by introducing a mechanistically-interpretable
framework for de novo protein binder engineering: integrating sparse
autoencoders (SAEs) with ESM-based protein models to extract
human-interpretable features associated with molecular recognition, and
using them to steer design toward high-affinity binding. The framework is
benchmarked against leading design architectures and, as proof-of-concept,
validated against clinically relevant stroke biomarkers -- **UCH-L1, S100B,
and B-FABP** -- via split-reporter survival assays. The goal is for
mechanistic interpretability to become a universal accelerant for rational
protein binder design, adaptable to any disease where early detection saves
lives.

**New to this repo and want to run the feature-labeling pipeline
yourself?** See [`FEATURE_LABELING_SETUP.md`](FEATURE_LABELING_SETUP.md)
for a step-by-step setup + run guide, including getting an Anthropic API
key and setting a spend limit. This README covers what the pipeline does
and why; that doc covers how to actually run it on your machine.

This directory (`sae/`) is the interpretability framework itself: SAE
training + benchmarking infrastructure for ESM-C-based binder embeddings.

## Status: validated on vilip1, not yet applied to the three clinical targets

The pipeline here was built and validated end-to-end on **vilip1** (VSNL1) --
a development/proof-of-pipeline target with an existing large Boltz
binder-design campaign in this repo (`data/vilip1_full20k/`,
`data/vilip1-design-composite_hotspot-20260724-v2/`), not one of SENTINEL's
three clinical targets. The reasoning: prove the SAE-training + benchmarking
methodology works and produces a sensible, defensible result on a target
where we already had a mature clustering/embedding pipeline
(`notebooks/bridget/esmc_embedding_analysis/`), before spending the same
effort on UCH-L1, S100B, and B-FABP.

Those three targets already have their own design campaigns in this repo
(`data/UCH_L1_final/`, `data/fabp7_full20k/` [B-FABP/FABP7],
`boltz-experiments/s100b-design-*/`) and clustering/embedding analysis
(`notebooks/bridget/esmc_embedding_analysis/esmc_*_uchl1.ipynb`,
`esmc_*_fabp7.ipynb`) -- applying this same SAE pipeline to them (same code,
different `--manifest`/`--data-dir`) is the natural next step, not yet done.

## Directory layout

```
sae/
  pretraining/
    inference/embed_esmc.py   # ESM-C embedding extraction (pooled + --per-residue mode)
    verify_sae_formula.py     # one-off: confirms Biohub's official SAE forward-pass math
                               # against their real installed source
  training/
    data.py                    # loads per-residue activations, builds train/val split
    sae_model.py                # our SAE: per-token TopK (k=64), mean-centering, AuxK dead-feature revival
    train.py                    # trains our SAE, logs per-epoch metrics + checkpoints
    benchmark.py                 # ours vs. Biohub's official general-purpose SAE, on the same held-out residues
    feature_analysis.py          # per-feature density + max-activating examples, linear probe vs. binding metrics
    label_features.py            # LLM auto-labeling of features from their max-activating examples (optional, costs API $)
    fetch_interpro.py            # InterPro domain/family annotations per example residue, via EBI's REST API (optional)
  analysis/
    sae_benchmark_analysis.ipynb   # training curves + benchmark comparison plots
    sae_feature_analysis.ipynb     # feature density/interpretation plots + probe results, reads feature_analysis.py's output
```

Data and outputs are gitignored (large/generated) and live locally / on the
cluster, not in this repo:

- `vilip1_layer23_per_residue/` -- `activations.npy` + `index.csv` +
  `manifest_combined.csv` (25,013 sequences: 20k full20k + 5k composite_hotspot
  + 13 natural binders, ~1.94M residues total).
- `vilip1_layer23_sae_outputs/checkpoints/run1/` -- our trained SAE
  (`best.pt`, `latest.pt`, `training_log.csv`).
- `vilip1_layer23_sae_outputs/benchmark_results/benchmark_summary.csv` --
  ours-vs-Biohub comparison.
- `vilip1_layer23_sae_outputs/analysis/` -- plots from the analysis notebook.

## Pipeline (as run for vilip1; same steps apply to any target)

**1. Combine manifests + build the training pool.** For vilip1: full20k (20k
designs) + composite_hotspot (5k designs) = training/val pool; 13 real
UniProt natural binders held out entirely for qualitative eval only (too few
points to be statistically meaningful for training, and structurally
different -- 84-815 residues vs. the designs' 50-150).

**2. Extract per-residue activations at one ESM-C layer** (GPU, on Waluigi):
```
python embed_esmc.py --manifest manifest_combined.csv --output <target>_layer23_per_residue/ \
    --layers 23 --per-residue
```
Layer 23 was chosen for vilip1 to match Biohub's own general SAE exactly
(direct comparability) and matched our earlier layer-sweep evidence across
fabp7/uchl1/vilip1 clustering work -- worth re-checking per-target rather
than assuming layer 23 is universally best.

**3. Train a target-specific SAE** (only needs torch/numpy/pandas, no
GPU-specific deps -- can run locally):
```
python train.py --data-dir <target>_layer23_per_residue --output-dir checkpoints/<run> \
    --d-hidden 16384
```
Architecture: per-token TopK (k=64, matching Biohub's sparsity so any
benchmark isn't confounded by a different sparsification scheme), dictionary
size originally set to the target's own token budget (4096 for vilip1's
~1.94M tokens, vs. Biohub's 16,384 trained on ~1000x more general data) but
since bumped to 16,384 to match Biohub's dict size directly -- a larger
dictionary plus k-annealing (below) measurably improved natural-binder
generalization FVE (0.42 -> 0.5157 in-house), fixed dataset-level
mean-centering (arXiv:2605.31518 -- prevents outlier-activation-dimension-
induced dead features), AuxK auxiliary loss for dead-feature resurrection.
Train/val split stratified per source so different design campaigns/length
regimes stay proportionally represented in both splits.

K-annealing (`--k-start`/`--k-anneal-frac`, default k-start=4x target k
decayed linearly over the first 20% of steps): starts training with a
looser top-k so every feature gets early gradient signal, then tightens to
the target k=64 -- see `sae_model.py`'s K-ANNEALING docstring. Matters more
at larger dictionary sizes, where more features compete for the same k
slots per token and a fixed small k from step one kills off more of them
before they specialize.

**4. Benchmark against Biohub's official general-purpose SAE** on identical
held-out residues:
```
python benchmark.py --checkpoint checkpoints/<run>/best.pt --manifest manifest_combined.csv \
    --output-dir benchmark_results
```
Encode uses Biohub's real official API (`model.add_sae_models(...)`), not a
guessed formula. Decode uses their published `W_dec`/`b_dec` weights and
their exact verified reconstruction formula (confirmed against their real
installed source, `transformers.models.esmc.modeling_esmc_sae._ESMCSAELayer`,
including the per-token z-score normalization step before `W_enc`/`W_dec`,
and the fact that `idf`/`max` are display-only stats, not used in
reconstruction). FVE is computed against a shared baseline (the trained
checkpoint's train-set mean) for both models, not a naive zero baseline --
ESM-C activations have large near-constant outlier dimensions that inflate
zero-baseline FVE regardless of whether a model captures anything
token-specific.

**5. Analyze**: plot training curves (loss/FVE/dead-feature-count vs. epoch)
and the benchmark comparison.

**6. Feature (semantic) analysis** -- what do the SAE's individual features
mean, and are any of them predictive of binder quality?
```
python feature_analysis.py --checkpoint checkpoints/<run>/best.pt \
    --data-dir <target>_layer23_per_residue --output-dir feature_analysis_results \
    --probe-metrics-csv <per-campaign manifest with binding_confidence/iptm/ipsae/ipae> \
    --probe-targets binding_confidence,iptm,ipsae,ipae
```
Encodes a stratified residue sample to find each feature's density (how
often it fires) and top max-activating residues with local sequence
context -- the standard SAE-interpretability recipe (Bricken et al. 2023,
"Towards Monosemanticity"): read the contexts around a feature's hardest-
firing residues to guess what it detects. Separately, pools per-residue
codes into one vector per protein (max- and mean-pool) and runs a linear
probe (per-feature Spearman correlation + nested-cross-validated Lasso)
against binding-quality metrics the SAE never saw during training -- a
feature that is both interpretable and predictive of binder quality is
real evidence of a usable structural correlate, not just a reconstruction
artifact. `sae_feature_analysis.ipynb` reads this script's output CSVs and
cross-references the two (predictive features -> their max-activating
contexts) for the actual biological read.

**7. (Optional) LLM auto-labeling** -- have a cheap model draft a one-
sentence description of each feature's pattern from its max-activating
examples, instead of a human reading every leaderboard by hand:
```
python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \
    --output-dir feature_analysis_results \
    --candidates-csv feature_analysis_results/probe_ipsae_max_multivariate.csv
```
Uses the Message Batches API (50% cheaper, appropriate for this bulk/non-
interactive job) with Claude Haiku 4.5 -- ~$0.05 for the probe-selected
candidate features, ~$3 for the full dictionary (4096 features). Requires
`pip install anthropic` and API billing (**not** covered by a Claude.ai
Pro/Max subscription -- that only covers claude.ai/Claude Code usage, not
programmatic API calls). Every label is a hypothesis from sequence
evidence alone, not a validated finding -- treat it the way you'd treat a
human's first-pass guess reading the same table, and confirm anything that
matters against structure before relying on it.

**Known limitation, partially addressed**: sequence context alone can miss
a feature whose real basis is structural (buried core, binder-target
interface) -- see `fetch_interpro.py` below for one mitigation. Raw 3D
structural context (secondary structure, solvent burial, coordinates) is
still deferred: Boltz's predicted structures (already in this repo's
`results/` directories, used by `compute_ipae.py`/`compute_ipsae.py` and
the `esmc_embedding_analysis` notebooks' structural sanity checks) carry
per-residue pLDDT and 3D coordinates that could add this, but (a) the
`results/` directory layout is per-campaign and wasn't available to build
against yet, and (b) raw coordinates are a poor fit for an LLM to reason
about compared to symbolic annotations -- `fetch_interpro.py` was built
instead, precisely to avoid needing an LLM to do spatial reasoning.

**8. (Optional) InterPro domain/family annotations** -- `fetch_interpro.py`
looks up each max-activating example's full protein via EBI's InterProScan5
REST API and cross-references any hit's position range against the
specific residue that fired the feature, so `label_features.py` can use a
real domain/family call as evidence instead of guessing from sequence
alone:
```
python fetch_interpro.py --examples-csv feature_analysis_results/feature_top_examples.csv \
    --manifest manifest_combined.csv --output-dir feature_analysis_results \
    --email you@example.com --candidates-csv feature_analysis_results/probe_ipsae_max_multivariate.csv

python label_features.py --examples-csv feature_analysis_results/feature_top_examples.csv \
    --output-dir feature_analysis_results \
    --candidates-csv feature_analysis_results/probe_ipsae_max_multivariate.csv \
    --interpro-csv feature_analysis_results/interpro_annotations.csv
```
**Expected coverage is sparse and that's not a bug**: this corpus is
overwhelmingly de novo Boltz-designed binders, not evolved natural
proteins. InterPro's member databases (Pfam, PROSITE, SMART, ...) are
profile HMMs built from evolutionary conservation across natural
homologs -- a de novo design has no evolutionary relationship to any
characterized family even when it successfully mimics a natural fold, so
most design-source examples are expected to come back with no hit.
Coverage should concentrate on the `natural_binders`/`binder_dataset_vilip1`
examples. `label_features.py`'s prompt is written to use an InterPro hit as
strong evidence when present and fall back to sequence-context-only
reasoning when absent, not to require a hit.

No local InterProScan install needed -- this hits EBI's free public REST
API directly (verified against the live service, not docs alone: one
job per unique protein sequence, no true batch endpoint). Requires
`pip install requests`, your own contact email (EBI's usage policy), and
is sized for a few dozen candidate proteins (e.g. the linear probe's
feature list), not the full ~25k-protein corpus -- each job takes several
minutes and this is a shared, free research service, not a bulk API.

## vilip1 proof-of-pipeline results (50-epoch run)

| | ours (dict=4096) | Biohub (dict=16384) |
|---|---|---|
| FVE, held-out designs (194,341 residues) | **0.941** | 0.716 |
| FVE, natural binders (5,362 residues, 13 proteins) | 0.420 | **0.702** |
| Dead features, held-out designs | 1/4096 (0.02%) | 1945/16384 (11.9%) |
| Dead features, natural binders | 29/4096 (0.7%) | 6522/16384 (39.8%) |

Domain-specific training clearly wins within-domain (higher FVE, far more
complete dictionary utilization) but at the expected cost of generalization
to real, out-of-domain proteins -- a specialist/generalist trade-off,
measured directly rather than assumed. This validates the pipeline is worth
applying to the actual clinical targets.

## Open next steps

- **Apply this same pipeline to UCH-L1, S100B, and B-FABP** -- the actual
  SENTINEL validation targets. Data/campaigns already exist for all three;
  this SAE pipeline does not yet touch them.
- **Run `feature_analysis.py` against the real 65k checkpoint** (the linear
  probe needs a per-campaign manifest with `binding_confidence`/`iptm`/
  `ipsae`/`ipae` joined in -- see `notebooks/bridget/esmc_embedding_analysis/`
  for how those get merged) and work through `sae_feature_analysis.ipynb`'s
  output: which features (if any) predict binder quality, and do their
  max-activating contexts correspond to legible structural motifs. The
  tooling for this (density + max-activating examples + probe) exists as of
  this commit but hasn't been run end-to-end on real data yet.
- Cross-check predictive/interpretable features against the HDBSCAN clusters
  from the earlier `esmc_embedding_analysis` clustering notebooks -- do SAE
  features and cluster membership agree on what makes designs similar?
