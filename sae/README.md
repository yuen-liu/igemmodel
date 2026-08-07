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
  analysis/
    sae_benchmark_analysis.ipynb   # training curves + benchmark comparison plots
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
- Linear probe: do either model's pooled per-protein sparse codes predict
  binder-quality metrics (`binding_confidence`/`iptm`/`ipsae`/`ipae`)? Tests
  whether features are functionally meaningful, not just good at
  reconstruction -- the actual "steer design toward high-affinity binding"
  piece of the SENTINEL mission.
- Cross-check against the HDBSCAN clusters from the earlier
  `esmc_embedding_analysis` clustering notebooks.
- Feature interpretation for whatever comes out of the probe as most
  predictive -- what residues/positions activate those features, and
  whether they correspond to known binding-relevant motifs.
