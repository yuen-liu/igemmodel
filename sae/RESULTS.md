# Vilip1 SAE results log

Running record of every training/benchmark run in the vilip1 SAE roadmap,
for pulling numbers into the New England Comp Bio conference abstract
later (https://newenglandcompbio.org/) without re-digging through chat
history. Update this whenever a new run/benchmark lands.

All runs: ESM-C layer 23, per-token TopK SAE (`sae/03_train/sae_model.py`),
`k=64`, `auxk=256`, `batch_size=4096`, `epochs=50`, `lr=4e-4`. "Natural FVE"
= `natural_binders_qualitative` pooled FVE from `benchmark.py` (ours vs.
Biohub's official ESMC-300M-sae-k64-codebook16384, both compared against
the same shared baseline). Biohub's own natural FVE is consistently
~0.70 regardless of our checkpoint (~0.6972-0.7036 across benchmark runs).
**Read the note directly below before quoting any "Natural FVE" number.**

## IMPORTANT (2026-09-26): the "natural binder" eval set is mostly NOT binders

The 82-sequence natural eval set (`EVAL_ONLY_SOURCES` in
`sae/02_prepare_data/data.py`) is two different things, and only one of
them is real binders:

| Source | n | Actually binds Vilip-1? | Trained on? |
|---|---|---|---|
| `natural_binders` (hand-picked, `data/VILIP-1_natural_binders.fasta`) | 13 | **yes** | never, in any run (`ALWAYS_EVAL_ONLY_SOURCES`) |
| `binder_dataset_vilip1` (STRING-derived, from `data/binder_dataset_raw.csv`) | 69 | **no** -- STRING `functional_or_physical_association` only, not validated binding | eval-only by default; 55/69 folded into training in run4/run6 |

Follow-up curation (reported 2026-09-26) established that the 69
STRING-derived sequences are not Vilip-1 binders. They are real human proteins
co-associated with VSNL1 in STRING (confidence 0.40-0.885), which is a
functional-association score, not evidence of physical binding. Only the
13 hand-picked UniProt sequences bind.

Consequences, in order of how much they matter:

1. **Every "Natural FVE" number below is dominated by non-binders.** The
   69 are 35,869 of 41,231 eval residues (87%) in the runs that hold out
   all 82 (run1/run2/run3/run5), since FVE is residue-weighted. Read that
   column as *reconstruction of natural human protein sequence*, NOT
   *reconstruction of Vilip-1 binders*.
2. **The 55 sequences mixed into training in run4/run6/run_paired (every
   `--natural-train-frac 0.8` run) were all non-binders.** Finding #2 below is still a real effect, but the
   mechanism is "mix in natural proteome sequence", not "mix in binders"
   -- see the rewritten finding.
3. **No binder-specific training leak.** Since `natural_binders` (the 13)
   is `ALWAYS_EVAL_ONLY`, no run ever trained on a real binder. A
   13-only FVE is therefore apples-to-apples across *every* row in the
   table, including run4/run6 -- which the pooled 82-vs-~27 column never
   was. See "Re-benchmark plan" below.
4. The NECB abstract (`necb_conference/necb_abstract.tex`) already says
   "natural-protein sequences" and "natural-protein reconstruction
   fidelity" rather than "binders" for the 0.39->0.60 result, so that
   claim holds as written. Do not reword it to say "binders".

The 13 real binders: O94919 (ENDOD1), P04155 (TFF1), Q03692 (COL10A1),
Q9BXJ5 (C1QTNF2), O43610 (SPRY3), Q7Z698 (SPRED2), Q99732 (LITAF),
Q86UW9 (DTX2), P54253 (ATXN1), Q86TD4 (SRL), Q7Z699 (SPRED1), Q96AQ9
(FAM131C), Q9BXU2 (TEX13B).

Counting note: `binder_dataset_raw.csv` has 70 VSNL1 rows, not 69.
Q96AQ9 (FAM131C) appears in both source files and was deduped out of the
STRING set, so it is counted in the 13, not the 69.

"Natural FVE (pooled)" is the number as originally reported: pooled over
whichever of the 82 were held out for that run, so 87%-by-residue
non-binder. "Binder-only FVE (13)" is the comparable-across-all-rows
number from the same `benchmark_summary.csv`'s `source=natural_binders`
row -- only run1's is in the repo; the rest need the re-benchmark below.

| Run | Training data | d_hidden | k-anneal | natural-train-frac | Design FVE (val, pooled) | Natural FVE (pooled, mostly non-binder) | **Binder-only FVE (13)** | Dead (natural eval) | Eval n (natural) = binders + non-binders |
|---|---|---|---|---|---|---|---|---|---|
| run1 (baseline, pre-session) | vilip1 65k | 4096 | no | 0 | 0.9656 | 0.3851 | **0.4404** | 1/4096 | 82 = 13 + 69 |
| run2_kanneal16384 (norm bug) | vilip1 65k | 16384 | 128->64 | 0 | 0.9715 | 0.4278 | not computed | 69/16384 | 82 = 13 + 69 |
| run3_normfix | vilip1 65k | 16384 | 128->64 | 0 | 0.9730 | 0.4156 | not computed | 65/16384 | 82 = 13 + 69 |
| run4_natural_mix | vilip1 65k | 16384 | 128->64 | 0.8 | 0.9660 | **0.5987** | not computed | 1017/16384 | ~27 = 13 + ~14 |
| run5_multi_target | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0 | 0.9723 | 0.4495 | not computed | 25/16384 | 82 = 13 + 69 |
| run6_combined | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0.8 | 0.9686 | 0.5973 | not computed | 683/16384 | ~27 = 13 + ~14 |
| run_paired (co-attention) | vilip1 65k, binder+target via ESM-C's native `\|` chain-break | 16384 | 128->64 | 0.8 | 0.9688 | not benchmarked vs. biohub (input-distribution mismatch, see below) | -- | -- | -- |
| Andrew's own run (external, not reproduced) | vilip1 65k + other-target mix (exact ratio unconfirmed) | 16384 | 128->64 (his own schedule) | unknown | -- | 0.5157 (reported) | not computed (his own eval set, "85 sequences" -- composition unverified) | -- | 82 (assumed) |

**Key findings, in order discovered:**
1. Decoder-norm axis bug (`w_dec` normalized per-output-dim instead of
   per-feature) found + fixed comparing against Andrew's pushed notebook --
   real bug, but retested (run3 vs run2) showed it did NOT explain the FVE
   gap (0.4156 vs 0.4278, a wash). Kept anyway -- mathematically correct.
2. **Natural-*sequence* mixing (`--natural-train-frac`) is the single
   biggest lever found** -- run4 alone (0.5987) beats Andrew's reported
   0.5157. Two caveats, the second added 2026-09-26:
   - run4/run6's natural FVE is measured on a smaller eval set (~27
     held-out, vs. 82 for every other row) since 55/69 STRING-derived
     sequences got folded into training -- not perfectly apples-to-apples,
     but the eval-set-size doesn't explain a jump this large.
   - **The 55 sequences folded into training are NOT Vilip-1 binders**
     (see the note at the top). So what this lever actually shows is that
     mixing *real human proteome sequence* into a mostly-synthetic design
     corpus improves reconstruction of held-out real protein sequence.
     That is still the biggest lever measured, and it is still a
     legitimate generalization result -- but it is not evidence about
     binders, and the eval set it is measured on is itself ~half
     non-binder (13 binders + ~14 non-binders). Describe it as
     natural-protein mixing, never as natural-binder mixing.
3. Multi-target diversity (run5 vs run3: 0.4495 vs 0.4156) helps on its
   own, but does NOT meaningfully stack with natural-sequence mixing
   (run6 vs run4: 0.5973 vs 0.5987, statistically a wash). Real
   side-effect: run6 has far fewer dead features than run4 (683 vs. 1017)
   despite the tied FVE -- larger/more diverse corpus keeps more of the
   dictionary alive.
4. Andrew's reported 0.5157 apparently came from a different run (not
   pushed to this repo) that also mixed in other-target binders --
   unconfirmed exact ratio/method.
5. `run_paired` (co-attention/binder+target via ESM-C's native chain-break
   token, confirmed present in `ESMCTokenizer().get_vocab()` as `|`) is
   NOT directly comparable to the others via `benchmark.py` -- that script
   re-runs ESM-C on the plain (binder-alone) sequence text, which doesn't
   match what run_paired's SAE was trained on (target-influenced
   activations). Its held-out design FVE (0.9688) is in line with every
   other run, so reconstruction quality isn't degraded. The actual
   co-attention question ("does it predict binding affinity better") is
   answered by the `feature_analysis.py` probe comparison, not FVE --
   see `sae/results/run4/` vs `sae/results/paired/` results once both are in.

**Decision (2026-08-13): `run4_natural_mix` chosen as the primary
checkpoint for feature analysis** (Vignesh's `feature_analysis.py`) --
tied with run6 on the metric that matters, simpler/cheaper, and analysis
tooling was already built around it.

**2026-08-14, ~1am: linear probe deprioritized for time.** Even after
parallelizing `LassoCV` (`n_jobs=-1`, commit `aaa6b3c`) across 48 cores, it
was still too slow to finish before the submission deadline allowed for --
switched to just the qualitative pass (`feature_analysis.py` without
`--probe-metrics-csv`: `feature_stats.csv` + `feature_top_examples.csv`,
under a minute). This means: no `cv_r2` numbers, and **the co-attention
question (does `run_paired` predict binding affinity better than
binder-alone) is NOT answered** -- that was the probe's job specifically,
nothing else in the pipeline tests it. If there's time before the actual
submission, revisit: profile why the parallel version was still slow
(worth checking whether joblib's process-based parallelism was spending
most of its time re-serializing the ~850MB pooled-code matrix to workers
rather than actually fitting), or just accept a much smaller/faster probe
(fewer alphas, fewer CV folds -- both hardcoded in `feature_analysis.py`,
not exposed via CLI) rather than skipping it entirely.

## Feature analysis results

**Qualitative pass done (2026-08-14), probe not run (see note above).**
`sae/results/run4/` and `sae/results/paired/` (scp'd down from Waluigi's
`~/analysis/`) each have `feature_stats.csv` + `feature_top_examples.csv`
from `feature_analysis.py` without `--probe-metrics-csv`.

- Both dictionaries healthy: 18/16384 dead (run4), 31/16384 dead (paired),
  both <0.2%. Similar density distributions (median ~0.0016-0.0018) and
  activation-magnitude distributions between the two.
- **Cross-dictionary consistency check**: took the top 30 highest-density
  (most generic) features in each of run4/run_paired, looked at each
  one's single hardest-firing residue (protein id + position). 6 of ~28-30
  landed on the EXACT same (protein, position) in both -- independently
  trained SAEs, different training data (binder-alone vs. binder+target),
  converging on the same residues by chance across millions of candidate
  residues would be essentially impossible. Real, non-artifactual signal.
  Several of the overlapping hits are on real natural-protein ids
  (Q9ULV0, Q9ULU8, Q03692, P49810), not just synthetic designs -- the
  shared signal isn't a design-campaign-specific artifact. **2026-09-26:
  of those four, only Q03692 (COL10A1) is a real Vilip-1 binder; Q9ULV0
  (MYO5B), Q9ULU8 (CADPS) and P49810 (PSEN2) are from the 69 STRING
  non-binders.** This finding is unaffected -- it is about two
  independently trained dictionaries converging on the same residues of
  the same real human proteins, which does not depend on those proteins
  binding anything. But do not restate it as "converges on binder
  residues"; the correct claim is "converges on real-protein residues".
- Single most generic feature in BOTH dictionaries peaks at the same
  residue: `binder_dataset_vilip1`, position 1715, context
  `MQLRYN[I]SQLEEW` (run4 feature 14247, density=0.7646; paired feature
  582, density=0.7631).
- **InterPro annotations (2026-08-14)**: ran `fetch_interpro.py` on an
  11-feature hand-picked subset per checkpoint (3 highest-density +
  3 rare-but-strong, per `FEATURE_LABELING_SETUP.md`'s own pilot recipe,
  plus one representative feature per cross-dictionary overlap hotspot
  above). Real domain/family coverage, surprisingly high for a mostly-de-
  novo-design corpus (expected sparse per the setup doc): run4 131/165
  examples annotated (79.4%), run_paired 136/165 (82.4%) -- makes sense in
  hindsight, since this feature subset specifically targets features tied
  to real natural-protein examples, not synthetic designs. Notable
  cross-checkpoint pattern: run4's feature 2214 fires on 9 consecutive
  real-protein positions (365-375 in evalbinder_P36222/P36222), all in
  `GH_hydrolase_sf (IPR017853)`; run_paired's feature 3896 shows the same
  coherent-motif pattern independently -- 5 consecutive positions
  (370-374 in evalbinder_P49810) all in `Peptidase_A22A (IPR001108)`.
  (2026-09-26: both P36222/CHI3L1 and P49810/PSEN2 are from the 69
  non-binders. The `evalbinder_` id prefix is just the manifest naming
  convention for `binder_dataset_vilip1` rows -- it does not mean the
  sequence was ever shown to bind. The domain-coherence result stands on
  its own; it's a statement about InterPro domains, not about binding.)
  Full results + methodology in
  `sae/notebooks/sae_feature_analysis_run4_vs_paired.ipynb`.
- **LLM auto-labeling done** (`label_features.py`, Claude Haiku 4.5 via
  Message Batches API, same 11-feature subset per checkpoint, InterPro
  evidence included). Well-calibrated: correctly returns "No clear
  pattern" when InterPro evidence is genuinely scattered (run4 feature
  1707; run_paired features 5997/9725) rather than forcing a story, and
  gives specific, confident labels when evidence is strong (run4 feature
  2214: "all 15 examples GH_hydrolase_sf"; feature 12918: tubulin/FtsZ
  GTPase domain + a specific recurring motif `TGLQG[F]L`).
- **Checked the shared-`--seed 0` confound on the cross-dictionary
  consistency finding**: both run4 and run_paired used the same default
  seed (identical weight init), so checked whether the SAME feature
  *index* is the hotspot representative in both dictionaries more than
  chance would predict. Only 1/6 hotspots share an index; the other 5 are
  detected by genuinely different features per dictionary -- the
  consistency finding is NOT mainly a seed artifact, a stronger result
  than assumed going in.
- **Probe (`cv_r2`) not run** -- see the "linear probe deprioritized"
  note above. Co-attention question still open.
- Full methodology + reproducible cells:
  `sae/notebooks/sae_feature_analysis_run4_vs_paired.ipynb`.

## Re-benchmark plan: binder-only (13-sequence) FVE

**Why:** the pooled "Natural FVE" column mixes 13 real binders with 69
non-binders, at 87%-by-residue non-binder, and its eval set changes size
between rows (82 vs ~27). The `source=natural_binders` row of the same
output is 13 real binders in every run, never trained on in any run --
the only natural-eval number that is comparable across the whole table.

**Good news: no code change needed.** `benchmark.py:239-241` already
accumulates per-source stats and writes a `source` column, so every
existing `benchmark_summary.csv` on Waluigi *already contains* the
binder-only row. run1's is in the repo:

| source | n_residues | ours FVE | biohub FVE | ours dead |
|---|---|---|---|---|
| `__pooled__` (82) | 41231 | 0.3851 | 0.7036 | 1/4096 |
| `binder_dataset_vilip1` (69 non-binders) | 35869 | 0.3795 | 0.7026 | 1/4096 |
| `natural_binders` (13 real binders) | 5362 | **0.4404** | 0.7134 | 50/4096 |

So step 1 is just to pull the CSVs that already exist, and only re-run
`benchmark.py` for any run whose CSV was lost.

**Step 1 -- check what's already on Waluigi (cheap, do this first):**

```bash
ls -la ~/notebooks/sae_training/benchmark_results_65k*/benchmark_summary.csv
for f in ~/notebooks/sae_training/benchmark_results_65k*/benchmark_summary.csv; do
  echo "=== $f"
  grep -E 'source|natural_binders' "$f"
done
```

Anything that prints a `natural_binders` row is already done -- no GPU
time needed, just scp it down.

**Step 2 -- re-run only for missing runs.** Per-run, substituting the
checkpoint dir (`run2_kanneal16384`, `run3_normfix`, `run4_natural_mix`,
`run5_multi_target`, `run6_combined`):

```bash
source /tmp/esm_verify_venv/bin/activate   # NOT esmfold2_venv -- meta-device crash
cd ~/notebooks/sae_training
RUN=run4_natural_mix
python benchmark.py \
    --checkpoint checkpoints_65k/$RUN/best.pt \
    --manifest manifest_combined.csv \
    --output-dir benchmark_results_65k_$RUN \
    --smoke-test
python benchmark.py \
    --checkpoint checkpoints_65k/$RUN/best.pt \
    --manifest manifest_combined.csv \
    --output-dir benchmark_results_65k_$RUN
grep -E 'source|natural_binders' benchmark_results_65k_$RUN/benchmark_summary.csv
```

Notes:
- run5/run6 were trained on the combined multi-target pool, so they need
  the combined manifest (the `combine_datasets.py` output used at train
  time), not vilip1's `manifest_combined.csv`. The natural rows are
  identical either way, but the `held_out_designs` split needs the
  manifest that actually contains their val ids.
- The 13 are only 5,362 residues, so this is fast -- the cost is
  re-running ESM-C + Biohub's SAE hook over the other sequences in the
  same pass. If GPU time is tight, the binder-only number alone can be
  had from a manifest filtered to `source == natural_binders`.
- Expect dead-feature counts on the 13-only row to look alarming
  (run1: 50/4096 ours, 6522/16384 biohub) -- that is the small-eval-set
  artifact already noted for run4, not a regression. Compare dead counts
  only between rows with the same eval set.

**Step 3 -- once the numbers are in:** fill the "Binder-only FVE (13)"
column above, and check whether the run3 -> run4 jump (0.4156 -> 0.5987
pooled) still holds on real binders only. If it shrinks a lot, the
0.39 -> 0.60 claim in the NECB abstract is specifically about natural
protein sequence, not binders, and the abstract's current wording
("natural-protein reconstruction fidelity") is already the right claim --
don't strengthen it. If it holds, that's a stronger result than what's
currently written.
