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

## Feature analysis results (fill in as they land)

- `feature_analysis_run4/`: probe of run4's pooled codes against
  `binding_confidence`/`iptm`/`ipsae` (no `ipae` data available -- not
  computed for any vilip1 campaign on Waluigi as of this session; `ipsae`
  only available for the `vilip1_full20k` subset, 20k/65k designs).
  cv_r2 per target/pooling: TBD.
- `feature_analysis_paired/`: same probe, run_paired's codes. cv_r2 per
  target/pooling: TBD. **This vs. the row above is the direct answer to
  the co-attention question.**
