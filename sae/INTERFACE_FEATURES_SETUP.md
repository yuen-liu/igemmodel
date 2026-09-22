# Interface feature mapping: setup & run guide

For the steering team: this tells you, for a set of SAE features you already
care about (e.g. the ones in `sae/results/run4/feature_labels.csv`), **which
of them actually fire near the real 3D binding interface of your designs**
-- and, for each one, whether it fires right at the interface or just nearby.
That's the input list you steer on: features that light up at the interface
are candidates for "this helps/hurts binding"; features that never do are
probably about something else (core packing, solubility, whatever) and
steering on them isn't testing a binding hypothesis.

If you just want to run it and get your YAML, skip to
[Step 2](#step-2-run-it). This doc is the mechanical "how do I run it"
companion; for *why* it's built this way, see the script's own docstring in
[`05_interpret/extract_interface_features.py`](05_interpret/extract_interface_features.py).

## What it does, in one picture

For every design (one predicted structure = one design):

```
target chain -------- binder chain
                        res 1   >10A from target  -> ignored
                        res 2   6A from target     -> "shell"   (near interface)
                        res 3   3A from target     -> "contact" (direct interface)
                        res 4   4A from target     -> "contact"
                        ...
```

Then for each feature you asked about: does it fire (nonzero SAE code) on
any residue that's a `contact` or `shell` residue? If yes, that design goes
in the output, tagged with exactly which residues, at which tier.

**`contact` and `shell` are always kept separate**, per feature per design --
never merged into one bucket -- so you can tell "fires right on the
interface" apart from "fires nearby" when you're deciding which features are
worth steering on:

```yaml
pres_ZQ4mh1P7FGiBS6xh3ISM:
  interface_features:
    233:
      tier: contact              # best tier this feature reached in this design
      residues:
        contact: [12, 13]        # <=5A from the target -- direct interface residues
        shell: [50]              # 5-10A from the target -- nearby, not direct contact
      max_activation: 8.42
```

Distances are measured straight to the target chain (not "near another
contact residue"), using the design's own predicted 3D coordinates -- not
ipTM/ipSAE's PAE-based method, which measures the model's *confidence*
about a pair of residues, not whether they're actually close in space. A bad
interface (exactly the kind you want "hurts binding" features from) can have
high PAE right where it matters, so a confidence-based definition would
tend to miss it.

## What you need

1. **A trained SAE checkpoint + its matching activations directory.** These
   two must be the exact matched pair (see
   [`FEATURE_LABELING_SETUP.md`](FEATURE_LABELING_SETUP.md)'s "What you need
   before starting" for why that matters and how to double check it). For
   vilip1, this already exists on the local machine used to build this
   script, outside the git repo (gitignored -- large files):
   ```
   sae/checkpoints/best.pt
   sae/vilip1_layer23_65k_per_residue/{activations.npy, index.csv, manifest_combined.csv}
   ```
   If you're on a different machine, ask whoever has these (Vignesh/bridget)
   for that same pair, or use your own checkpoint + its own matching
   activations directory.
2. **A folder of predicted structures** for the designs you want checked --
   PDB or mmCIF. Either layout works, no need to reorganize anything:
   - a flat folder of files named by design id (`pres_XXXXX.pdb`, etc.), or
   - the same nested Boltz results-dir tree you already point
     `compute_ipsae.py`/`compute_ipae.py` at (`.../<design>/files/result/*_predicted.cif`
     + `metadata.json`) -- if that's what you use for ipSAE/ipAE, use the
     exact same folder here.
3. **A list of features to check.** Easiest options (pick one):
   - `sae/results/run4/feature_labels.csv` -- the features that already have
     a human/LLM-drafted label (small list, ~14 features as of this writing).
   - a `probe_*_multivariate.csv` or `probe_*_univariate.csv` from
     `feature_analysis.py`'s linear probe -- the features statistically
     associated with a binding metric (ipsae/ipae/binding_confidence).
   - or just type feature ids by hand if you know exactly which ones you want.

## Step 0: environment

Same environment as the rest of `05_interpret/` (see
`FEATURE_LABELING_SETUP.md`'s Step 0), plus one more package:

```bash
pip install torch numpy pandas pyyaml
```

## Step 1: know your paths

You'll need three things on hand before running the command below:

- `--checkpoint`: path to `best.pt` (see "What you need" #1 above)
- `--data-dir`: path to the matching activations directory (same #1)
- `--structures-dir`: path to your folder of predicted structures (#2) --
  wherever your Boltz results for this campaign actually live (this is
  campaign-specific and not something this doc can hardcode for you)

## Step 2: run it

From `sae/05_interpret/`:

```bash
cd sae/05_interpret
python3 extract_interface_features.py \
    --checkpoint /path/to/checkpoints/best.pt \
    --data-dir /path/to/vilip1_layer23_65k_per_residue \
    --structures-dir /path/to/your_boltz_results_folder \
    --candidates-csv ../results/run4/feature_labels.csv \
    --output interface_features.yaml
```

For vilip1, with the local matched pair from "What you need" #1, that's:

```bash
cd sae/05_interpret
python3 extract_interface_features.py \
    --checkpoint /Users/vigneshkarthik/Documents/liuLab/igemmodel/sae/checkpoints/best.pt \
    --data-dir /Users/vigneshkarthik/Documents/liuLab/igemmodel/sae/vilip1_layer23_65k_per_residue \
    --structures-dir /path/to/your_boltz_results_folder \
    --candidates-csv ../results/run4/feature_labels.csv \
    --output interface_features.yaml
```

(Swap `--candidates-csv` for `--features 233,1707,995` if you'd rather type
exact ids instead of pointing at a CSV.)

It prints progress as it goes (structures classified, then designs
processed) and finishes with a one-line summary:

```
1,842/20,000 designs had >=1 requested feature firing within 10A of the interface
14 designs skipped, e.g.: [('pres_abc123', 'id not found in --data-dir activations pool'), ...]
Wrote interface_features.yaml
```

Runtime: the structure-parsing step is CPU-only and parallelizes across
`--workers` (defaults to all cores); the SAE-encoding step is fast (small
per-design forward passes). Tens of thousands of designs should finish in
minutes, not hours.

## Step 3: read the output

```yaml
metadata:
  target_chain: A
  binder_chain: C
  contact_cutoff_angstrom: 5.0
  shell_cutoff_angstrom: 10.0
  features_checked: [233, 1707, 995]
  n_designs_with_interface_features: 1842
  n_designs_skipped: 14

designs:
  pres_ZQ4mh1P7FGiBS6xh3ISM:
    n_binder_residues: 87
    interface_features:
      233:
        tier: contact
        residues: {contact: [12, 13], shell: [50]}
        max_activation: 8.42

skipped:
  pres_abc123: "id not found in --data-dir activations pool"
```

To go from this to "which features should we steer on": for each feature id,
count how many designs it shows up in (and at which tier) across the whole
YAML -- a feature that only ever shows up as `shell`, never `contact`,
across hundreds of designs is a weaker binding-interface signal than one
that's consistently a direct `contact` residue. Cross-reference against
`feature_labels.csv`'s text label and, if you have it, the probe's
correlation direction (does firing correlate with *better* or *worse*
ipsae/ipae?) to decide which bucket ("helps binding" vs "hurts binding") a
feature belongs in.

## Troubleshooting

- **Lots of designs in `skipped` with "id not found in --data-dir activations
  pool"** -- normal if your structures folder includes designs outside the
  campaign your SAE was trained/interpreted on (e.g. a newer batch not yet
  embedded). Only designs already in `manifest_combined.csv` can be checked
  without re-running ESM-C embedding, which this script doesn't do.
- **A design skipped with "binder-chain sequence from structure does not
  match manifest sequence"** -- the script refuses to guess here rather than
  risk mismatching residues to the wrong SAE codes. Usually means
  `--binder-chain`/`--target-chain` don't match this structure's actual
  chain ids (default is target=`A`, binder=`C`, matching this campaign's
  Boltz spec -- pass `--target-chain`/`--binder-chain` if yours differ).
- **"No .pdb/.cif structures found"** -- check `--structures-dir` actually
  contains `.pdb`/`.cif` files somewhere under it (recursive search), or
  that the nested-layout files really are named `*_predicted.cif` under
  `files/result/`.
- **A design has zero interface features but you expected some** -- check
  `n_binder_residues` in its entry looks right (roughly matches the design's
  known length) and that your feature ids are actually in
  `metadata.features_checked` -- a typo'd id there silently checks nothing
  for that id rather than erroring.
