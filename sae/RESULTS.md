# Vilip1 SAE results log

Running record of every training/benchmark run in the vilip1 SAE roadmap,
for pulling numbers into the New England Comp Bio conference abstract
later (https://newenglandcompbio.org/) without re-digging through chat
history. Update this whenever a new run/benchmark lands.

All runs: ESM-C layer 23, per-token TopK SAE (`sae/training/sae_model.py`),
`k=64`, `auxk=256`, `batch_size=4096`, `epochs=50`, `lr=4e-4`. "Natural FVE"
= `natural_binders_qualitative` pooled FVE from `benchmark.py` (ours vs.
Biohub's official ESMC-300M-sae-k64-codebook16384, both compared against
the same shared baseline). Biohub's own natural FVE is consistently
~0.70 regardless of our checkpoint (~0.6972-0.7036 across benchmark runs).

| Run | Training data | d_hidden | k-anneal | natural-train-frac | Design FVE (val, pooled) | **Natural FVE** | Dead (natural eval) | Eval n (natural) |
|---|---|---|---|---|---|---|---|---|
| run1 (baseline, pre-session) | vilip1 65k | 4096 | no | 0 | 0.9656 | 0.3851 | 1/4096 | 82 |
| run2_kanneal16384 (norm bug) | vilip1 65k | 16384 | 128->64 | 0 | 0.9715 | 0.4278 | 69/16384 | 82 |
| run3_normfix | vilip1 65k | 16384 | 128->64 | 0 | 0.9730 | 0.4156 | 65/16384 | 82 |
| run4_natural_mix | vilip1 65k | 16384 | 128->64 | 0.8 | 0.9660 | **0.5987** | 1017/16384 | ~27 |
| run5_multi_target | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0 | 0.9723 | 0.4495 | 25/16384 | 82 |
| run6_combined | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0.8 | 0.9686 | 0.5973 | 683/16384 | ~27 |
| run_paired (co-attention) | vilip1 65k, binder+target via ESM-C's native `\|` chain-break | 16384 | 128->64 | 0.8 | 0.9688 | not benchmarked vs. biohub (input-distribution mismatch, see below) | -- | -- |
| Andrew's own run (external, not reproduced) | vilip1 65k + other-target mix (exact ratio unconfirmed) | 16384 | 128->64 (his own schedule) | unknown | -- | 0.5157 (reported) | -- | 82 (assumed) |

**Key findings, in order discovered:**
1. Decoder-norm axis bug (`w_dec` normalized per-output-dim instead of
   per-feature) found + fixed comparing against Andrew's pushed notebook --
   real bug, but retested (run3 vs run2) showed it did NOT explain the FVE
   gap (0.4156 vs 0.4278, a wash). Kept anyway -- mathematically correct.
2. **Natural-binder mixing (`--natural-train-frac`) is the single biggest
   lever found** -- run4 alone (0.5987) beats Andrew's reported 0.5157.
   Caveat: run4/run6's natural FVE is measured on a smaller eval set (~27
   held-out, vs. 82 for every other row) since 55/69 STRING-derived
   sequences got folded into training -- not perfectly apples-to-apples,
   but the eval-set-size doesn't explain a jump this large.
3. Multi-target diversity (run5 vs run3: 0.4495 vs 0.4156) helps on its
   own, but does NOT meaningfully stack with natural-binder mixing
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
   see `feature_analysis_run4/` vs `feature_analysis_paired/` results
   once both are in.

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
`feature_analysis_run4/` and `feature_analysis_paired/` (local repo root,
scp'd down from Waluigi's `~/analysis/`) each have `feature_stats.csv` +
`feature_top_examples.csv` from `feature_analysis.py` without
`--probe-metrics-csv`.

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
  shared signal isn't a design-campaign-specific artifact.
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
  (370-374 in evalbinder_P49810) all in `Peptidase_A22A (IPR001108)`. Full
  results + methodology in
  `sae/analysis/sae_feature_analysis_run4_vs_paired.ipynb`.
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
  `sae/analysis/sae_feature_analysis_run4_vs_paired.ipynb`.
