# SENTINEL

Columbia iGEM's project on mechanistically-interpretable de novo protein
binder design. Stroke remains a leading cause of disability and death
worldwide, with therapeutic efficacy declining rapidly post-onset.
Circulating biomarkers offer a path to early diagnosis, but designing
high-affinity biosensing binders is constrained by the opacity of
generative protein models -- they operate as black boxes, obscuring the
features that actually drive high affinity and specificity.

SENTINEL addresses this by integrating sparse autoencoders (SAEs) with
ESM-based protein models to extract human-interpretable features associated
with molecular recognition, and using them to steer design toward
high-affinity binding. The framework is benchmarked against leading design
architectures and, as proof-of-concept, validated against clinically
relevant stroke biomarkers -- **UCH-L1, S100B, and B-FABP** -- via
split-reporter survival assays.

Concretely, this repo does two things: (1) runs Boltz-based binder-design
campaigns against each target and scores the results (interface metrics,
structure prediction), and (2) trains/benchmarks/interprets sparse
autoencoders on ESM-C embeddings of those designs, to find human-readable
features that correlate with binding quality.

## New here? Start with what you're trying to do

- **Working on the SAE interpretability pipeline** (training, benchmarking,
  feature interpretation)? Go to [`sae/README.md`](sae/README.md) -- it has
  its own full walkthrough, numbered folder-by-folder.
- **Running a Boltz design campaign for a target**, or scoring one? See
  [Design campaigns](#design-campaigns--data) below for where campaign data
  lives and [`scripts/`](#scripts) for the scoring tools.
- **Working on the cluster (Waluigi)**? See [Cluster](#cluster) below.
- **Looking for the conference submission**? See [`conference/`](#conference).

## Repo map

```
sae/                # SAE training + benchmarking + interpretability pipeline (see sae/README.md)
data/                # per-target Boltz design-campaign outputs (gitignored -- see below)
boltz-experiments/   # raw Boltz design-experiment configs + run outputs (gitignored -- see below)
notebooks/bridget/   # per-person exploratory notebooks (design experiments, embedding/clustering analysis)
scripts/             # shared analysis utilities: interface-metric scoring, clustering, result ranking
cluster/waluigi/      # launch scripts for the shared Waluigi GPU box (no SLURM -- run directly, in tmux)
conference/          # New England Comp Bio (NECB) submission: abstract + one-pager, source + compiled PDF
```

## Design campaigns & data

`data/` holds per-target Boltz binder-design campaign outputs -- designed
sequences, structures, and interface-quality metrics (`ipae.csv`,
`manifest_with_ipsae.csv`, `embeddings*.npz`, `clustering/`, `results/`).
One subdirectory per campaign, e.g. `data/UCH_L1_final/`,
`data/fabp7_full20k/` (B-FABP/FABP7), `data/vilip1_full20k/` +
`data/vilip1-design-composite_hotspot-*/` (vilip1/VSNL1, the SAE pipeline's
proof-of-concept dev target -- see `sae/README.md`'s "Status" section for
why), `data/REG3A_*` (a non-clinical diversity target, used only to broaden
the SAE's multi-target training pool -- not one of SENTINEL's three
clinical targets).

`boltz-experiments/` holds design-experiment configs (`payloads/*.yaml`)
and their raw run outputs, organized by design strategy rather than by
target (`crf-design-*`, `s100b-design-*`).

**Both `data/` and `boltz-experiments/` are entirely gitignored** -- they're
large (multi-GB per campaign) and regenerable from Boltz runs, not meant to
be committed. If you clone this repo fresh, these directories will be
empty; ask the team for a copy of whichever campaign(s) you need, or
regenerate them with the `boltz-*` skills / cluster scripts.

`notebooks/bridget/` has per-person exploratory work: `esmc_embedding_analysis/`
(ESM-C embedding + HDBSCAN clustering analysis per target -- the earlier
work the SAE pipeline builds on), `crf_binder/` and `s100b_binder/`
(design-strategy experiments), `boltz-experiments/` (raw run outputs tied
to those notebooks). Only the `.ipynb` files themselves are tracked; the
raw `boltz-experiments/` subfolder here is gitignored like the top-level one.

## Scripts

Shared, target-agnostic analysis utilities in `scripts/`:

- `compute_ipsae.py` / `ipsae.py` -- interface predicted aligned error
  score (ipSAE), using the vendored `ipsae.py` (DunbrackLab/IPSAE reference
  implementation, called as a subprocess rather than reimplemented, to
  avoid transcription bugs in a metric used to filter real candidates).
- `compute_ipae.py` -- mean interface predicted aligned error (ipAE),
  BindCraft/ColabDesign convention -- not the same metric as Boltz's own
  `min_interaction_pae`.
- `cluster_dbscan.py` -- DBSCAN clustering over ESM-C embeddings with
  top-scoring representative selection per cluster.
- `rank_boltz_results.py` -- ranks a downloaded Boltz campaign and writes a
  top-N manifest, from either a `results/` directory or an `index.jsonl`.

## Cluster

`cluster/waluigi/` holds launch scripts for Waluigi, the shared GPU box:

- `run_embed_esmc.bash` -- runs ESM-C embedding extraction (see `sae/01_embed/`).
- `start_jupyter_boltzgen.bash` / `start_jupyter_esmfold2.bash` /
  `start_jupyter_genie3.bash` -- set up (first run) and launch a Jupyter
  server for each of three binder-design tools being evaluated (BoltzGen,
  ESMFold2-based design, Genie3).

Waluigi has **no SLURM scheduler** -- run scripts directly, ideally inside
`tmux`/`screen` so they survive disconnects. Its 2 GPUs are shared with
another user, so check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`
explicitly rather than assuming a free GPU. Each tool gets its own isolated
venv/conda env (`boltzgen_venv`, `esmfold2_venv`, a `genie3` conda env,
`/tmp/esm_verify_venv` for the SAE benchmark's real-`esm`-package
requirement) -- there's no single repo-wide environment or requirements
file; read the relevant `start_jupyter_*.bash`/script header before running
anything for the first time on a machine you haven't set up yet.

## Conference

`conference/` has the New England Comp Bio (NECB) submission materials --
`necb_abstract.tex`/`.pdf` and `necb_onepager.tex`/`.pdf` -- built from the
vilip1 SAE proof-of-pipeline results (`sae/RESULTS.md`).
