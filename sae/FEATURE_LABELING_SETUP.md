# Feature labeling pipeline: setup & run guide

Step-by-step instructions for running the feature interpretation + labeling
pipeline (`feature_analysis.py` -> `fetch_interpro.py` -> `label_features.py`)
on your own machine. For *why* this pipeline works the way it does (what
"semantic analysis" means for an SAE, why residues not proteins, why
InterPro coverage is expected to be sparse for most designs, etc.), see
[`README.md`](README.md) -- this doc is the mechanical "how do I actually
run it" companion, not a replacement.

## What you need before starting

Two data artifacts, downloaded from wherever the team keeps them (ask
Vignesh/bridget), **as a matched pair**:

1. A trained SAE checkpoint (`best.pt`).
2. The exact activations directory that checkpoint was trained on --
   `activations.npy` + `index.csv` + `manifest_combined.csv` together in
   one directory.

**These two must be a matched pair, not just any checkpoint + any
activations directory.** The checkpoint stores `mean`/`scale` computed
from one specific activations pool at training time, and its
`val_protein_ids` reference that pool's protein-id space. If you pair a
checkpoint with a *different* activations directory, nothing will crash --
it'll just silently produce meaningless numbers. If you're not sure which
activations directory matches your checkpoint, check that the checkpoint's
`train_protein_ids`/`val_protein_ids` are a subset of the directory's
`index.csv` ids, and that the `source` composition matches what you expect
(ask whoever gave you the checkpoint what run it's from). A short way to
check from Python:

```python
import torch
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
print(ckpt["config"])          # d_model, d_hidden, k -- should match what you expect
print(ckpt["val_fve"])         # cross-check against any published number for that run
print(len(ckpt["val_protein_ids"]) + len(ckpt["train_protein_ids"]))  # total design proteins
```

## 0. Environment setup

Any Python 3.9+ environment works (a project conda env, a fresh venv --
doesn't matter, as long as you install into it and run from it
consistently). No GPU needed -- everything here runs comfortably on a
laptop CPU.

```bash
# from inside your chosen environment
pip install torch numpy pandas requests anthropic
```

(`torch`/`numpy` for step 1, `pandas` for all three scripts, `requests`
for step 3, `anthropic` for step 5.)

Place your downloaded files somewhere like:
```
sae/checkpoints/best.pt
sae/data-dir/activations.npy
sae/data-dir/index.csv
sae/data-dir/manifest_combined.csv
```
(These paths are already gitignored -- see the repo's `.gitignore` -- so
you can drop multi-GB files here without worrying about accidentally
committing them.)

## Step 1: Run the base interpretation pass

```bash
cd sae/training
python3 feature_analysis.py \
    --checkpoint ../checkpoints/best.pt --data-dir ../data-dir \
    --output-dir ../results
```

Takes well under a minute on CPU. Writes two files to `sae/results/`:
- `feature_stats.csv` -- one row per feature: density, fire count, mean
  activation when active, dead flag.
- `feature_top_examples.csv` -- for each feature, its top-15 hardest-firing
  residues with local sequence context.

## Step 2: Pick which features to look at

You don't need to (and shouldn't, at first) run every feature in the
dictionary -- start with a small, deliberately mixed sample rather than the
whole thing. A good first pilot is 2-3 high-density features (broad,
probably-generic signal) plus 2-3 rare, strongly-firing features (narrow,
probably-specific signal) -- comparing the two tells you whether the
pipeline is actually distinguishing signal from noise, not just producing
plausible-sounding text regardless of input.

```python
import pandas as pd
stats = pd.read_csv("../results/feature_stats.csv")
alive = stats[~stats["dead"]]

print("Most common (likely generic):")
print(alive.sort_values("density", ascending=False).head(5)[["feature", "density", "fire_count"]])

print("Rare but strong (likely specific):")
rare = alive[(alive["fire_count"] >= 20) & (alive["fire_count"] <= 200)]
print(rare.sort_values("mean_activation_when_active", ascending=False).head(5)[["feature", "density", "fire_count", "mean_activation_when_active"]])
```

Note the feature ids you want (comma-separated, no spaces, e.g.
`1803,995,318,675,1902`) -- you'll reuse this exact list in every step
below.

## Step 3: Fetch InterPro domain/family annotations (optional but recommended)

Cross-references each example residue's exact position against real
domain/family database annotations (via EBI's free InterProScan REST API),
giving the labeling step something more concrete than sequence text alone
to reason from.

```bash
python3 fetch_interpro.py \
    --examples-csv ../results/feature_top_examples.csv \
    --manifest ../data-dir/manifest_combined.csv \
    --output-dir ../results \
    --email your_own_email@example.com \
    --features 1803,995,318,675,1902
```

- `--email` is required by EBI's usage policy -- use your own address, not
  a shared one.
- This submits one real job per unique protein and polls until done --
  expect several minutes of wall-clock wait for a handful of features, not
  instant. It'll print progress as it polls.
- **Expect most examples to come back with no annotation** -- this dataset
  is mostly de novo designed proteins with no direct evolutionary
  relationship to characterized families, so sparse coverage is normal, not
  a sign anything's broken. That said, don't be surprised if you *do* see
  a recurring hit across many different proteins at a consistent position --
  that's a sign the design campaign shares a common scaffold fold, and it's
  a genuinely informative result if you see it (we saw this ourselves:
  several features turned out to consistently land in a Ubiquitin-like
  scaffold domain at the same relative position across dozens of designs --
  worth actually reading the `interpro_annotation` column yourself, not
  just trusting a summary of it).

Writes `sae/results/interpro_annotations.csv`.

## Step 4: Preview the labeling prompt (free, no API key needed)

```bash
python3 label_features.py \
    --examples-csv ../results/feature_top_examples.csv \
    --output-dir ../results \
    --features 1803,995,318,675,1902 \
    --interpro-csv ../results/interpro_annotations.csv \
    --dry-run
```

Prints the exact prompt that would be sent, for one feature, with no API
call and no cost. Read it and make sure it looks sane before spending
anything. (Skip `--interpro-csv` here if you didn't run step 3.)

## Step 5: Get an API key, set a spend limit, then run for real

This step costs real money -- cheap (a handful of features costs well
under $0.10 on Haiku 4.5), but real. Do the account setup first.

### 5a. Set a spend limit (do this before creating a key)

1. Go to [platform.claude.com](https://platform.claude.com) and log in.
2. If you have more than one Anthropic organization, check which one
   you're in (org switcher, top-left / account menu). If you only have
   one, skip this check.
3. Go to **Settings -> Billing**, find **Spend limits**, click **Set
   limit** (or **Adjust limit**), enter a number (e.g. `5` for $5), save.

This is an account-wide monthly cap -- once total spend on that org hits
the limit, API usage pauses until next month (or until you raise it). It's
a real backstop, not just a warning.

### 5b. Create an API key from that same org

Right after 5a, **without switching orgs or logging out**, go to
**Settings -> API Keys** in the same browser tab -- this guarantees the
key you create is under the org you just capped. Click **Create Key**,
name it something recognizable (e.g. `sae-feature-labeling`), and copy the
value (`sk-ant-...`) -- it's only shown once.

### 5c. Export it in the same terminal/environment you'll run from

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Important: this only applies to the shell session it's run in. If you're
using a conda environment, activate that environment *first*, then export
the key in that same terminal, then run the script from that same
terminal -- exporting it somewhere else (a different terminal tab, a
different environment) won't be visible when you actually run the script.

Quick check it's really set:
```bash
echo ${ANTHROPIC_API_KEY:0:12}...
```

### 5d. Run it for real

Same command as step 4, minus `--dry-run`:

```bash
python3 label_features.py \
    --examples-csv ../results/feature_top_examples.csv \
    --output-dir ../results \
    --features 1803,995,318,675,1902 \
    --interpro-csv ../results/interpro_annotations.csv
```

Writes `sae/results/feature_labels.csv` -- one row per feature, with its
LLM-drafted one-sentence label.

## Reading the output

Every label is a hypothesis from the evidence you gave it, not a validated
finding -- read it against the raw evidence in `feature_top_examples.csv`
(and `interpro_annotations.csv` if you ran step 3) yourself before trusting
it. A feature with a tight, consistent local sequence motif *and*
independent InterPro/structural corroboration (like feature 318 in our own
pilot run) is much more convincing than one where the label sounds
plausible but the underlying evidence is scattered.

| File | What it is |
|---|---|
| `feature_stats.csv` | density / fire count / dead flag per feature |
| `feature_top_examples.csv` | top-15 max-activating residues per feature, with sequence context |
| `interpro_annotations.csv` | InterPro hit (if any) per example, at that exact residue position |
| `feature_labels.csv` | final LLM-drafted one-sentence label per feature |

## Troubleshooting

- **`python3: can't open file '.../label_features.py'`** -- you're not in
  `sae/training/`. `cd` there first; all the commands above assume that's
  your working directory (they use `../results`, `../data-dir`, etc.).
- **Checkpoint loads but results look wrong/nonsensical** -- double check
  the checkpoint/data-dir pairing (see "What you need before starting"
  above). This is the most likely silent-failure mode.
- **`fetch_interpro.py` seems stuck** -- normal, EBI jobs take several
  minutes each. It prints a poll count as it waits; if it's still climbing,
  it's working, not stuck.
- **API key "not found" / `anthropic.AuthenticationError`** -- almost
  always an environment mismatch: the key was exported in a different
  terminal/environment than the one running the script. Re-export it in
  the exact terminal you're about to run the command from.
