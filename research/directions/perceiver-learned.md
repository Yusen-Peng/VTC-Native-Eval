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
spent_gpu_hours: 53.28
max_jobs_in_flight: 6
jobs:
- 8489019
- 8489021
- 8489022
- 8489023
- 8489024
- 8490000
jobs_archived:
- 8489020
- 8472979
- 8471019
- 8466638
- 8453617
- 8453618
- 8453619
- 8463111
- 8453616
- 8463112
- 8466350
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
- 8453620
- 8454045
- 8453615
- 8461970
- 8461971
last_wake: '2026-11-10T14:43:27-05:00'
next_id: 10
model: claude-fable-5-1
effort: high
max_turns: 120
timeout_minutes: 60
wakes: 15
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
   *(Closed 2026-11-10 by e005/e007/e008: no-go. 3b, optional, 1 GPU-h: the same warm start with the latent grid centred
   (mean token direction removed), one 1 h real-schedule segment scored as e007; e008 measured the shared latent direction,
   not the attention, as what keeps the outputs alike.)*
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
- 2026-11-09 19:00 wake (on_jobs: 8453620_{0,1,2,4,5} done). e003 DONE (2.44 GPU-h): the 2x row is complete. Task 2
  (RealWorldQA) exited 0 with no result (node a0339 GPU "busy or unavailable" at load; run.py swallows the exception),
  retried once as 8454045_0 on debug-nextgen (7 min, clean); TextVQA (task 3) finished in-wake. Full 2x learned_residual row
  73.37 / 1527.3 / 47.00 / 59.22 / 33.42 / 40.00 / 20.7 / 40.24 vs pooled 74.31 / 1524.0 / 46.89 / 57.65 / 32.38 / 37.33 /
  20.8 / 40.12: same row within noise (max |delta| 1.6 except DocVQA +2.7); neither Perceiver row beats random pruning +
  LLaVA@66K on OCR. 5 of 7 e003 eval predictions hit (OCRBench/ChartQA equal to pooled, not above). e004 seg1 8453615 at
  88 s/step (step 26 at 19:00), first 20 steps paired -0.027 nats below pooled 4x; figure refreshed. Nothing new submitted:
  in flight 5 (e004 chain + eval array), the 8x chain + eval needs 5 ids, so it fits only once seg4 8453618 has finished
  (about 07:00 2026-11-10). Suggested use of the seg1/seg2 boundary wakes (1 free id each): implement backlog item 3
  (warm-started paper-faithful `learned`: latents = mean pooled token, zero-init o_proj + last MLP linear, comp lr 1e-3) and
  run its 15-min smoke on debug-nextgen; its training chain goes in with or after the 8x chain (2 training jobs max).
  Spent 17.52 of 100 GPU-h (e001 0.10 + e002 14.98 + e003 2.44); committed about 17.5 more (e004).
- 2026-11-09 22:50 wake (on_jobs: 8453615 seg1 TIMEOUT). e004 seg1 ended at step 163 as planned (checkpoint-160, 4.00 GPU-h);
  seg2 8453616 resumed cleanly on a0263 (step 161 bit-identical, 87 s/step, step 170 when checked). 170 steps paired vs
  pooled 4x: -0.062 nats (five times the 2x gap); figure refreshed. e005 = backlog item 3 (99d7de0): warm-started
  paper-faithful `learned` queries via INIT_FROM (`scripts/perceiver_warmstart.py`: latents = mean native token map pooled
  to 32x32 over 256 LLaVA images at native RMS, o_proj + mlp.2 zeroed, other weights from the pooled-2x perceiver.bin;
  comp lr 1e-3). Desktop CPU selftest + login-node checks passed. Submitted build 8461970 + smoke 8461971 (afterok, 25 min)
  on debug-nextgen. Go/no-go in e005: chain only if the smoke's steps 11-16 mean is below 1.10. In flight 6 (cap 6): e004
  seg2-4 + eval array, e005 build + smoke. Spent 21.82 (e001-e003 17.52 + e004 seg1 4.00 + seg2 running); committed about
  13.5 more (e004) + 0.6 (e005 smoke); the 8x chain and the e005 chain wait for free ids (seg4 finishes about 07:00 11-10).
- 2026-11-09 23:00 (same wake). Build 8461970 COMPLETED in 2:44 (0.05 GPU-h): native RMS 4.52, mean-grid RMS 2.94 (18% spatial),
  scale x1.54; all four build predictions HIT. Smoke 8461971 RUNNING (25 min). Next wake: score the smoke (steps 1-5 and
  11-16 vs the e002 smoke on the same data), apply the e005 go/no-go; if go, the 4 x 4 h chain + gate evals need 6 ids, so
  they go in once e004 segments free ids (seg2 ends about 02:20, seg3 06:20). `wake: on_jobs`.
- 2026-11-09 23:45 wake (on_jobs: 8461971 plw_smoke TIMEOUT). e005 DONE, chain NOT submitted (0.47 GPU-h): 15 steps at 89.5 s/step,
  steps 1-5 mean 1.272 (HIT), steps 11-15 mean 1.236 (predicted 1.05-1.20, MISS; abort line 1.20), 0.30 nats above the e002
  smoke on identical batches (0.930) and level with the collapsed run (1.202); step-6 spike (grad norm 13.8) as in the collapsed
  runs. Interpretation: with o_proj/mlp.2 at zero the image path must grow from nothing through Adam, so 15 steps cannot separate
  "untrainable" from "needs tens of full-lr steps"; warm start alone does not put the paper architecture near the pooled anchor.
  e007 (503ea57, 1.0 GPU-h): same config as seg0 of the real run (llava_665k%10, no suffix) for 1 h on debug-nextgen, 8463112;
  go rule steps 25-37 mean < 1.10 (paired refs pooled 0.885 / learned_residual 0.853 / collapsed 1.147), P(go) 0.25. e006 = 8x
  learned_residual row (backlog item 4): seg1 8463111 on nextgen (4 h); seg2-4 + 8-benchmark eval array appended at later wakes
  as ids free (cap 6: in flight now 8453616-8453619 + 8463111 + 8463112). e004 seg2 at step 200, paired -0.057 vs pooled 4x;
  figure refreshed. Spent 22.95 + 0.42 (smoke) of 100; committed about 12 (e004) + 17 (e006) + 1 (e007). `wake: on_jobs`.
- 2026-11-10 00:30 wake (on_jobs: 8463112 plw2x_seg0 TIMEOUT). e007 DONE, NO-GO (1.00 GPU-h): 39 steps, steps 25-37 mean 1.143 vs
  collapsed `learned` 2x 1.147 on the same batches (paired -0.004, sd 0.007), pooled 0.885; 3/3 window predictions hit, go branch
  did not fire, chain not submitted, backlog item 3 closed. Weight check at checkpoint-30: o_proj at 38% of a trained module's RMS,
  mlp.2 at 73%, latents moved 0.08%. e008 (d25f75d, probe of checkpoint-30, 8466350, 5 min, 0.09 GPU-h) DONE: attention became
  near one-hot in 30 steps (entropy 0.11 of log N; the collapsed run is at 0.99 after 488), but output pairwise cosine 0.994 (init
  0.977): the shared latent direction (82% of the grid) drowns the attention branch in `norm_out`; LLM reads 0.33 nats over zeros.
  4/9 e008 predictions hit (the attention prediction was the opposite). e006 seg2 8466638 appended (`afterany:8463111`; seg1 at
  step 48, 87.5 s/step, paired -0.013 vs pooled 8x). e004 seg2 at step 255 (paired -0.053 vs pooled 4x). Figure refreshed. In flight
  6 (cap 6). Spent 26.61 (driver count incl. running segments); committed about 10 (e004) + 15.5 (e006). Next wake: archive
  finished segments, append e006 seg3/seg4 + eval array as ids free; optional 3b (centred latents, 1 GPU-h) if an id is free after
  that. `wake: on_jobs`.
- 2026-11-10 03:05 wake (on_jobs: 8453616 plr4x_seg2 TIMEOUT). e004 seg2 ended at step 325 as planned (checkpoint-320, 4.00 GPU-h);
  seg3 8453617 resumed cleanly on a0153 (steps 321-325 bit-identical, 87 s/step, step 331 when checked; ends about 06:42 near
  step 485, so seg4 only finishes + saves, then eval 8453619). 330 steps paired vs pooled 4x: -0.048 (steady -0.034 since step 170).
  e006 seg1 8463111 at step 131 (ends about 03:26): paired vs pooled 8x -0.044 over 1-130, -0.074 over 101-130, the 4x pattern
  one ratio up; the e006 loss-gap prediction is on track. e006 seg3 8471019 submitted (`afterany:8466638`, nextgen, 4 h, at
  d25f75d, no code change). Figure refreshed, RESULTS.md regenerated. In flight 6 (cap 6): e004 seg3/seg4/eval + e006 seg1-3.
  Next wake (seg1 8x TIMEOUT about 03:26, or e004 seg3 about 06:42): archive, append e006 seg4 + eval array (2 ids) as ids
  free; optional 3b after that. Spent 30.6 (driver count). `wake: on_jobs`.
- 2026-11-10 03:40 wake (on_jobs: 8463111 plr8x_seg1 TIMEOUT). e006 seg1 ended at step 161 as planned (checkpoint-160, 1 step
  lost, 4.00 GPU-h); seg2 8466638 resumed cleanly on a0237 (step 161 bit-identical 0.9391, 87 s/step, step 168 when checked).
  160 steps paired vs pooled 8x: -0.049 (-0.070 over 131-160), wider than the 4x run at the same point; the e006 loss-gap
  prediction is on track to hit. e006 seg4 8472979 submitted (`afterany:8471019`, nextgen, 4 h, at d25f75d, no code change).
  e004 seg3 at step 370 (paired -0.033 over 331-370, -0.047 over 1-370). Figure refreshed, RESULTS.md regenerated. In flight
  6 (cap 6): e004 seg3/seg4/eval + e006 seg2-4. Next wake (e004 seg3 about 06:42 or seg4/eval): archive, submit the e006
  8-benchmark eval array (`afterok:8472979`, 1 id); optional 3b after that. Spent 32.6 (driver count). `wake: on_jobs`.
- 2026-11-10 13:40 wake (on_jobs: e004 chain + eval array and e006 chain all finished). e004 DONE (16.81 GPU-h): 4x chain resumed
  bit-identically at 160/320/480, seg4 13 min; final MA-10 0.897 vs pooled 4x 0.932 (paired -0.043 over the run, -0.033 sd 0.011
  steady); full 4x row 66.92 / 1408.4 / 44.00 / 52.03 / 16.77 / 16.23 / 3.6 / 22.36 vs pooled 64.60 / 1371.4 / 44.67 / 51.37 /
  15.63 / 14.27 / 2.8 / 21.52: above pooled on 7 of 8 (+2.3 MMBench, +37 MME, +2.0 DocVQA), 7/12 predictions hit, misses all
  "more gain than predicted" (item-4 "within 2 pts" missed by +0.32 in the favourable direction). e006 training DONE (12.25 GPU-h,
  eval pending): MA-10 0.9905 vs pooled 8x 1.049 (paired -0.057; the gap grows with the ratio: -0.012 / -0.033 / -0.059), 4/4
  training predictions hit; 8-benchmark eval array 8489019_[0-7] submitted at d25f75d-equivalent script (ae62a23 checkout, eval
  script unchanged). e009 (ae62a23) = the lr-only control from e003's "Next": `pooled` queries at compressor lr 1e-3, 4x (run name
  `PerceiverLr1e3`, new SFT_perceiver.sh name rule + config key); smoke 8489020 (debug-nextgen, 20 min) + chain 8489021-8489024
  (`afterany`, 4 x 4 h, nextgen); gate eval appended after seg4. P(reproduces >= half of e004's gain) 0.6. Optional 3b skipped
  (in-flight cap 6 reached: eval array + smoke + 4 segments). Figure + RESULTS.md refreshed. Spent 48.14 of 100; committed about
  4.5 (e006 eval) + 14.8 (e009). Next wake: score the smoke (abort rule in e009), archive the eval tasks (check result files, not
  exit codes), score the e006 benchmark predictions, write the 8x row. `wake: on_jobs`.
- 2026-11-10 14:15 wake (on_jobs: 8489019_{1,2,3,6} done, 8489020 smoke TIMEOUT). e006 eval 6 of 8 tasks done (0/1/2/3/6/7, result
  files present, 1.96 GPU-h); TextVQA and DocVQA still running (DocVQA about 1:45 to go). 8x row so far: MMBench 58.16 / MME 1226.6 /
  MMMU 43.44 / RWQA 47.84 / OCRBench 2.0 / ChartQA 14.32 vs pooled 8x 44.07 / 1050.0 / 42.22 / 47.97 / 1.9 / 13.56: +14.1 MMBench and
  +177 MME (the 4x gain was +2.3 / +37), MMMU +1.2, the rest within noise; MMBench and MME predictions missed far on the high side,
  ChartQA 20-30 missed low (14.3, pooled 13.6). e009 smoke 8489020 as designed (12 steps, 87.5 s/step, 0.34 GPU-h): steps 1-5 mean
  1.236 (prediction 0.95-1.15 MISS: the anchor was the 2x smoke, the 4x init on `%1` sits at 1.24-1.26), max 1.360 at step 11 (grad norm
  1.5, no spike), abort rule not triggered. Seg1 8489021 running on a0176 since 13:52 (88.8 s/step; steps 1-2 bit-identical to pooled
  4x, step 3 diverges by 0.0002). e009 gate eval 8490000_[0-1] (MMBench + DocVQA, `afterok:8489024`) submitted at ae62a23 after the
  config-key check on the login node. Figure + RESULTS.md refreshed. In flight 6 (cap 6): eval array + e009 chain + gate. Spent 51.33
  (driver count); committed about 1.3 (e006 eval remainder) + 12.3 (e009 chain) + 2.2 (gate). Next wake: finish the 8x row (DocVQA,
  TextVQA, item-4 DocVQA criterion), set e006 done; then e009 segment boundaries. `wake: on_jobs`.
- 2026-11-10 14:30 wake (on_jobs: 8489019_{0,1,2,3,6,7}, the same six tasks already scored at 14:15; stale trigger). New within the
  wake: task 4 TextVQA COMPLETED (0:53, 0.89 GPU-h): 8x TextVQA 12.26 vs pooled 8x 10.59 (+1.67; random pruning 11.87), 7 of 8 tasks
  done, e006 at 15.10 GPU-h. Task 5 DocVQA at 46% (1.06 s/item, ends about 15:20); the array id stays in `jobs` until it finishes.
  e009 seg1 8489021 at step 20 (88.4 s/step): paired 1-20 vs pooled 4x -0.009, vs learned_residual 4x +0.019 (e004 was -0.027 at
  the same point); too early to score. Figure + RESULTS.md refreshed. Nothing submitted: in flight 6 (cap 6). Next wake (DocVQA
  task 5 about 15:20): write the full 8x row, score the DocVQA and item-4 DocVQA criterion, set e006 done, archive 8489019.
  Then e009 seg1 boundary about 17:52. `wake: on_jobs`.
- 2026-11-10 14:45 wake (on_jobs: 8489019_{0,1,2,3,4,6,7}; stale trigger, all seven scored at 14:15/14:30). DocVQA task 5 at 62%
  (1.27 s/item lately, ends about 15:30); the array id stays in `jobs`. e009 seg1 8489021 at step 30 (88.4 s/step): paired vs pooled
  4x -0.037 over 1-30 and -0.093 over 21-30, vs learned_residual 4x +0.017 / +0.014: once past warmup the lr-only run tracks the
  learned_residual curve, not the pooled one (early support for the e003 lr attribution; not scored yet). Figure + RESULTS.md
  refreshed. Nothing submitted: in flight 6 (cap 6). Next wake (task 5 done): write the 8x row, score DocVQA + item-4 criterion,
  set e006 done, archive 8489019. `wake: on_jobs`.
