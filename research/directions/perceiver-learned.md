---
slug: perceiver-learned
title: Why does the learned-query Perceiver collapse to chance under LoRA SFT, and
  what initialisation makes it match the pooled-query variant at 2x?
project: vtc
type: direction
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 100
spent_gpu_hours: 15.14
max_jobs_in_flight: 6
jobs:
- 8453615
- 8453616
- 8453617
- 8453618
- 8453619
- 8453620
jobs_archived:
- 8387204
- 8387205
- 8387554
- 8387555
- 8387802
- 8387804
- 8387805
- 8387806
- 8387807
- 8387808
- 8387809
- 8387862
- 8387863
- 8387864
- 8387865
- 8387866
- 8387867
- 8387868
- 8453614
last_wake: '2026-11-09T18:20:00-05:00'
next_id: 5
model: claude-fable-5-1
effort: high
max_turns: 120
timeout_minutes: 60
wakes: 5
---
# Why does the learned-query Perceiver collapse, and what fixes it?

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs. `jobs` = ids in flight; finished ids move to
`jobs_archived` (never delete ids). `budget_gpu_hours` is a hard stop.

## Goal
`compressors/perceiver.py` has two query modes. `pooled` (queries = fixed-pooling outputs, tanh-gated residuals
starting at 0, so init = fixed pooling) trained and evaluated fine: 74.31 MMBench / 37.33 DocVQA at 2x, the
"+ LLaVA@66K, 2x, FP" row of `tab:training-based`. `learned` (the paper-faithful variant: a learned [D, 32, 32]
latent grid resized to the output grid, final LayerNorm rescaled to the native-token RMS) trained to completion
on the same recipe without any error but collapsed:

| run (LLaVA@66K, 488 steps) | final train loss | MMBench | MME | DocVQA | OCRBench |
|---|---|---|---|---|---|
| Perceiver pooled 2x / 4x / 8x | 0.76 / 0.91 / 1.03 | 74.31 / 64.60 / 44.07 | 1524 / 1371 / 1050 | 37.33 / 14.27 / 9.78 | 20.8 / 2.8 / 1.9 |
| Perceiver learned 2x / 4x / 8x | 1.09 / 1.10 / 1.09 | 24.23 / 24.05 / 26.98 | 875 / 858 / 843 | 6.85 / 7.29 / 7.01 | 1.7 / 1.7 / 1.5 |

Checkpoints: `/fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval-ckpts/output/NEO_continue_SFT_LLaVA_OV_llava66k_PerceiverLearned_{2,4,8}x`
(commit 52a7543 for 2x, 97dc1b5 for 4x/8x; `trainer_state.json` has the loss curves; jobs 8142084, 8250433,
8250434). Eval outputs: `VLMEvalKit/outputs/NEO-2B-SFT-OV-LLaVA-llava66k-PerceiverLearned-{2,4,8}x/` in
`$SCRATCH/VTC-Native-Eval`. The learned loss curve is the same at all three ratios (1.35 -> 1.09) and never
reaches the pooled 8x curve, which is the signature of an LLM that has learned to ignore its visual tokens.

Settled when a learned-query variant reaches, at 2x on LLaVA@66K, MMBench >= 72 and DocVQA >= 30 (within 3 and 7
of the pooled variant), with the cause of the collapse diagnosed by a measurement, not a guess. Secondary: if
the fix works at 2x, 4x and 8x rows for the `tab:training-based` Perceiver block.

## Theory
Three candidate causes, in the order the backlog tests them.
(a) **Signal, not optimisation:** the LLM's loss is ratio-independent, so the compressed tokens carry no usable
information from step 0 and the LoRA (lr 2e-5, 488 steps) settles into the text-only floor; the compressor's
gradient then vanishes (nothing downstream reads its output). A pooled-query module starts as fixed pooling,
which the LLM already partly understands (70.6 MMBench untrained), so the gradient to the adapter is informative
from the start. The paper's Perceiver trains end-to-end from scratch with a full model and far more steps; a
LoRA on a frozen LLM cannot afford a random visual front end.
(b) **Scale at init:** latents are trunc-normal std 0.02, then `norm_q` LayerNorm blows them up to unit RMS, so
the queries are random unit vectors; the output passes through `norm_out` and is rescaled to the native-token
RMS, so the LLM receives full-magnitude noise at every visual position. The pooled mode starts at gate 0 and
produces exactly the pooled tokens.
(c) **Learning rate:** compressor lr 1e-4 may be too small to move a from-scratch cross-attention block in 488
steps (DRIP's boundary predictor uses 1e-3). This alone is unlikely to be the cause (the loss would still fall
below the floor eventually), but it may be part of the fix.
Counter-argument to the whole direction: with 66K samples and a frozen LLM, no from-scratch resampler may be
trainable, and the honest table entry is "learned queries: fails at this data scale". That outcome is also a
result; record it as such if backlog items 2-4 all fail.

## Hypothesis backlog
1. **Diagnose (no training; `debug-nextgen` or desktop `gpu-run`, NEO-2B bf16 fits 16 GB).** On 64 LLaVA@66K
   samples, measure the LLM loss under: (i) the trained PerceiverLearned-2x adapter + compressor, (ii) the same
   adapter with the visual tokens replaced by zeros, (iii) the trained Perceiver-pooled-2x adapter + compressor.
   Also dump, for the trained learned compressor: attention entropy over keys per query (vs uniform), pairwise
   cosine similarity of its output tokens, and output RMS vs native-token RMS. Prediction (confidence 0.75):
   (i) and (ii) are within 0.02 nats (the LLM ignores the visual tokens), (iii) is at least 0.2 nats lower;
   output tokens have mean pairwise cosine > 0.9 (near-identical queries) or attention entropy within 5% of
   uniform. Write this as a script `scripts/perceiver_probe.py` so later variants are measured the same way.
2. **Pooled-anchored learned queries (cheapest fix, tests cause (a)+(b)).** New query mode
   `learned_residual`: queries = pooled tokens (as in `pooled`) + learned latent grid (bilinear-resized, init 0),
   keep the tanh-gated residuals from the pooled mode, so init = fixed pooling exactly, but the latents are free
   to move the queries away from the pooled content. Smoke on `debug-nextgen` (`DATASETS=llava_665k%1`,
   `SUFFIX=_smoke`, 15 min): loss at step 5 must be below 1.0 (the pooled curve), then full 2x training
   (12 h) and the gate eval (MMBench + DocVQA, about 2.3 GPU-h). Prediction (0.6): MMBench 73-75, DocVQA 35-40,
   i.e. matches pooled; the latents' norm after training tells whether "learned" adds anything (predict < 10% of
   the pooled-token norm, meaning the mode is redundant with pooled).
3. **Warm-started paper-faithful variant (tests whether the paper's architecture can work with a non-random
   start).** Keep `learned` queries but initialise the latent grid from the mean fixed-pooling output of 256
   LLaVA images (projected to the 32x32 grid), copy attention/MLP weights from the trained pooled-2x
   `perceiver.bin`, zero-init `o_proj` and the last MLP linear so the first forward is the pure (RoPE-positioned)
   query grid, and raise compressor lr to 1e-3 with the LoRA at 2e-5. Smoke, then full 2x + gate eval.
   Prediction (0.4): MMBench 68-73, DocVQA 25-35; if the step-5 smoke loss is above 1.2 abort and record.
4. **Only if 2 or 3 passes the gate:** full 8-benchmark eval of the passing variant at 2x, then train and
   evaluate 4x and 8x with the same variant (2 x 12 h + 2 x 4.5 GPU-h). Prediction (0.5): the learned variant
   is within 2 pts of pooled at every ratio on MMBench and within 5 on DocVQA; no variant beats ToMe on OCR.
5. **Only if 2 and 3 both fail the gate:** one run of plain `learned` with compressor lr 1e-3 and 3x the steps
   (`llava_665k%30`, chain of 9 x 4 h segments, resume from `checkpoint-*`) to test cause (c) in isolation.
   Prediction (0.3) that it crosses the pooled-8x loss curve. If it does not, pause with the "fails at this
   data scale" conclusion.

## Constraints
- Files the agent may change: `VLMEvalKit/vlmeval/vlm/neo/local_models/compressors/perceiver.py` (new query
  modes only; `pooled` and `learned` forward passes must stay bit-identical, the pooled rows are final and the
  `learned` checkpoints must still load with `strict=True`); `VLMEvalKit/vlmeval/VLMTrainKit/SFT_perceiver.sh`
  (new `QUERIES` values, `INIT_FROM`, `STEPS`/`DATASETS` knobs); `neo/train/build.py` (warm-start loading)
  and `neo/train/argument.py` (new args; defaults unchanged); `trainer.py` only if a new parameter group is
  needed (do not touch `DRIP_LR` or the DRIP groups); `neo_chat.py` only to pass a new `perceiver_queries`
  value through; new config keys in ONE block at the end of `neo_series` in `config.py`
  (`NEO-2B-SFT-OV-LLaVA-llava66k-PerceiverLearnedResidual-{r}x` etc.); `scripts/flops.py` gets a row for any
  new mode (`perceiver_<mode>`); `scripts/perceiver_probe.py` (new); `scripts/eval_table2.sbatch` and
  `SFT_perceiver.sh` may read `REPO=${REPO:-$SCRATCH/VTC-Native-Eval}`; `research/` records.
- Files that may NOT change: `fixed_pooling.py`, `drip*`, `modeling_*.py`, existing config keys, the existing
  Perceiver / PerceiverLearned checkpoints and outputs (read only), the tex `VTC-Native-Eval.tex` at the repo
  root (read it for `tab:training-based`; report rows in the record, the user pastes them).
- Fixed protocol: recipe of `SFT_perceiver.sh` (base `Paranioar/NEO1_0-2B-SFT`, `llava_665k%10`, LoRA lr 2e-5,
  global batch 128 on 1 GPU, 488 steps, warmup 10; 2 GPUs is untested and failed on 2026-10-28, jobs 8142086/8).
  Compressor lr is a variable of this direction; record it. Gate eval = MMBench_DEV_EN + DocVQA_VAL via
  `scripts/eval_table2.sbatch` with `MODELS="<key>"` (array tasks 0 and 5 of the two suites, or an explicit
  `DATASETS` override); full eval = both suites. Metrics as in `summarize_table2.py`. Compare against the pooled
  rows above and the untrained fixed-pooling 2x row (70.62 MMBench / 28.85 DocVQA).
- Per-experiment ceiling: training 1 A100-40GB, measured 11.4 h per 488 steps. Training jobs are chains of 4 h `nextgen` segments: `--time=4:00:00`, each segment resubmitted with `--dependency=afterany:<prev>` (3 segments cover the measured 11.4 h; the trainer resumes from the latest `checkpoint-*`, saved every 10 steps = 14 min, so at most 14 min is lost per boundary). Submit the whole chain at once and record every segment id.
  Smokes on `debug-nextgen`
  (15 min); probes on `gpu-run` or `debug-nextgen`. At most 2 training jobs in flight. Budget plan: probe 0.5 +
  2 smokes 0.5 + up to 3 trainings 36 + gates 7 + one full eval 4.5 + 4x/8x 32 = about 80 GPU-h.
- OSC: run from `/fs/scratch/PAS2836/jacktaylor/wt/vtc/perceiver-learned`; checkpoints in the shared
  `$SCRATCH/VTC-Native-Eval-ckpts/output/` under new run names (never overwrite the six existing ones).
- Every record: the hypothesis it tests (a/b/c above), prediction with confidence, the smoke loss at step 5,
  the probe numbers, and a PNG of training loss vs step for every run of this direction overlaid on the six
  existing curves (from `trainer_state.json`).

## Log
- 2026-11-07 14:05 wake (manual). Created the OSC checkout `wt/vtc/perceiver-learned` (clone of origin). e001 =
  backlog item 1 (diagnostic probe, no training): `scripts/perceiver_probe.py` + sbatch; import/tokenizer check passed
  on the login node from /tmp; submitted smoke 8387204 (N=4) and full probe 8387205 (N=128, afterok) on debug-nextgen
  at 94091f4. Baseline loss-curve PNG drawn (`figures/loss_curves.png`, `scripts/perceiver_curves.py`). Note for item 2:
  `SFT_perceiver.sh` still cds into `$SCRATCH/VTC-Native-Eval` (the old checkout) and `vlmeval` is an editable install
  pointing there; training from this worktree needs the `REPO` override before any training submission.
- 2026-11-07 14:10 e001 done (0.10 GPU-h, both jobs exit 0). Diagnosis = cause (b): the learned compressor emits S
  copies of one vector (output pairwise cosine 0.9953 at init, 0.9999 after training; attention uniform to 1% of
  log N) because 0.02-RMS latents are drowned by the O(1) attention residual; the LLM still reads 0.27 nats from
  that global vector vs zeros, so cause (a) in its strong form is refuted. Pooled gates end at 0.004 (the pooled
  Perceiver is fixed pooling + 0.4%). Next wake: item 2 (`learned_residual`), and item 3 gets the specific fix
  (zero-init `o_proj` + last MLP linear, latents at native RMS). Nothing in flight; `wake: now`.
- 2026-11-07 14:32 wake (manual). e002 = backlog item 2: `learned_residual` query mode (pooled queries + zero-init 32x32
  latent grid sampled at segment centroids, tanh gates; init == fixed pooling, verified bit-identical `pooled`/`learned`
  by `scripts/test_perceiver_modes.py` on the desktop and the login node). Compressor lr 1e-3 (recorded as a confound
  with the pooled row's 2e-5). `SFT_perceiver.sh` and `eval_table2.sbatch` now take `REPO`; a segment that finds the
  final perceiver.bin exits 0 (chain no-op). Submitted smoke 8387554 (25 min, to checkpoint-10) + resume smoke 8387555
  (12 min, afterany) on debug-nextgen at 7cca724. Next wake: if steps 1-5 average in [0.9, 1.1] and the resume log
  shows step 11, submit the 4 x 4 h `nextgen` chain + gate evals (about 16 GPU-h); else diagnose.
- 2026-11-07 15:20 wake (on_jobs: 8387554 TIMEOUT). Smoke 8387554 as planned: 16 steps, mean 1-5 = 0.979 (HIT), max 1.094,
  checkpoint-10 complete, curve sits on the pooled 2x curve (figure regenerated). Resume smoke 8387555 FAILED at once:
  `train.py` resume branch used `logging.info` on `transformers.utils.logging` (no such attribute); never-exercised path.
  Fixed (3085d93), resume smoke retried as 8387802 with the chain 8387804-8387807 + gates 8387808/8387809 behind it.
  8387802 failed in 48 s for a second, different cause: HF `_load_from_checkpoint` wants full model weights because the
  PeftModel is nested under `language_model`; the chain died with it (8387804 DependencyNeverSatisfied, the rest
  scancelled). Fixed in 43311d8 (`trainer.py` loads the adapter via `set_peft_model_state_dict`; a deviation from the
  trainer.py constraint, recorded in e002, DRIP groups untouched). Third resume smoke 8387862 + replacement chain
  8387863 (`afterok:8387862`) -> 8387864 -> 8387865 -> 8387866 (4 h each, nextgen, about 460 pending ahead), gates
  8387867 MMBench + 8387868 DocVQA (`afterok:8387866`). Resume smoke 8387862 PASSED within the wake: adapter (480
  tensors) + perceiver.bin loaded, steps 11-12 bit-identical to the uninterrupted smoke (1.0942, 0.8142); seg1 released
  from the `afterok` (a TIMEOUT smoke would have cancelled it), smoke scancelled at 4 min. Spent 0.65 GPU-h; e002 planned
  total about 16. Next wake: on the chain segments (check "Resumed LoRA adapter" + step at each boundary, refresh the
  loss figure), then the gate evals.
- 2026-11-07 15:25 wake (on_jobs: 8387555 FAILED). Stale trigger: 8387555 was diagnosed, fixed (3085d93) and archived in
  the 15:20 wake. State check: seg1 8387863 PENDING (Priority) on nextgen, 8387864-8387866 and gates 8387867/8387868
  PENDING (Dependency); no job finished, nothing new to record, nothing submitted (2 training jobs would exceed the
  constraint and the chain already covers the plan). RESULTS.md regenerated (unchanged). `wake: on_jobs`.
- 2026-11-09 18:20 wake (on_jobs: e002 chain + gates finished). e002 DONE: 4 x 4 h chain resumed cleanly at every boundary
  (163 / 323 / 487 steps, seg4 finished step 488 in 13 min; 87 s/step), final MA-10 loss 0.787 vs pooled 0.795 (paired, same
  data order), gate MMBench 73.37 / DocVQA 40.00 vs pooled 74.31 / 37.33: GATE PASSED (>= 72 / >= 30), all e002 predictions
  hit; 14.98 GPU-h. Figure regenerated. e003 (12a1540): probe 8453614 on the trained checkpoint ran in-wake (0.06 GPU-h):
  latents inert (0.2% of the query, 0.001 nats), attention gate 1e-4 with near one-hot attention, MLP gate 0.010; the whole
  33%-RMS change from fixed pooling is the per-token MLP branch, worth 0.031 nats; the e002 gain over pooled is the higher
  compressor lr, not learned queries. Remaining six 2x benchmarks submitted as one array 8453620_[0-5] (DATASETS override in
  eval_table2.sbatch). e004 = 4x row: chain 8453615-8453618 (4 x 4 h, nextgen) + full 8-benchmark eval 8453619_[0-7]
  (afterok seg4). In flight 6 (cap 6); 8x waits for the next wake. Spent 15.14 of 100 GPU-h; committed about 21 more.
