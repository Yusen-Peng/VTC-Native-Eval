---
slug: sft-4x8x
title: Do the training-based 2x conclusions hold at 4x and 8x? (fixed pooling, random
  pruning and DRIP + LLaVA@66K rows of tab:training-based)
project: vtc
type: direction
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 120
spent_gpu_hours: 67.59
max_jobs_in_flight: 12
jobs:
- 8489185
- 8489188
- 8489189
- 8489190
- 8489191
- 8489192
- 8489193
- 8489195
- 8490212
- 8490213
jobs_archived:
- 8489181
- 8489187
- 8489180
- 8489186
- 8454544
- 8454545
- 8454547
- 8454548
- 8464684
- 8464685
- 8463948
- 8464719
- 8470989
- 8470990
- 8474120
- 8474121
- 8454552
- 8463947
- 8454543
- 8454546
- 8454549
- 8454550
- 8454551
- 8463470
- 8463471
- 8462191
- 8462192
- 8453773
- 8453777
- 8453771
- 8453772
- 8453774
- 8453775
- 8453776
- 8453778
- 8453779
- 8453780
- 8454364
- 8454365
- 8454476
- 8454477
- 8388118
- 8388119
- 8388120
- 8388121
last_wake: '2026-11-10T15:25:00-05:00'
next_id: 12
model: null
effort: null
wakes: 12
---
# Do the training-based 2x conclusions hold at 4x and 8x?

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs. `jobs` = ids in flight; finished ids move to
`jobs_archived` (never delete ids). `budget_gpu_hours` is a hard stop.

## Goal
Fill the empty 4x and 8x cells of `tab:training-based` in `VTC-Native-Eval.tex` (2B scale) for the three
learning-based methods that have a 2x row but no 4x/8x row:

| row | 2x result (already in the table) | 4x | 8x |
|---|---|---|---|
| fixed pooling + LLaVA@66K | 74.39 MMBench / 37.44 DocVQA | empty | empty |
| random pruning + LLaVA@66K | 72.68 / 41.95 | empty | empty |
| DRIP + LLaVA@66K, two-tailed 5:5 | 75.08 / 80.04 | empty | empty |

Perceiver (pooled queries) 4x/8x rows are already filled (64.60/14.27 and 44.07/9.78); do not rerun them.
The question each cell answers: at 2x, continual SFT on LLaVA@66K recovers general VQA for fixed/random pooling
(+4 MMBench) but not OCR (DocVQA stays below 45), while DRIP with two-tailed inference recovers OCR to 80 DocVQA.
Settled when all six rows exist, each with all 8 benchmarks, so the tex can state whether (a) the SFT gain on
general VQA grows, shrinks or vanishes as compression increases, and (b) DRIP keeps its OCR advantage over
fixed/random at 4x/8x and how it compares with the best training-free method at the same ratio
(ToMe: 69.67 MMBench / 67.01 DocVQA at 4x, 57.99 / 38.48 at 8x).

## Theory
Fixed pooling and random pruning destroy fine text structure; SFT teaches the LLM to read the pooled embedding
space but cannot restore characters, so OCR stays flat and the gain concentrates on general VQA. At 4x/8x there
is less to recover for general VQA from an adapter alone (training-free fixed 4x is already at 41 MMBench), so
either the recovered gap grows (more headroom) or SFT at 488 steps is too short to move a 40-point hole.
DRIP selects boundaries by content; two-tailed selection keeps both the most and least boundary-like tokens,
which at 2x acted as a learned mix of keep-and-merge. At 4x/8x the boundary predictor must drop 75-88% of tokens;
if the learned boundaries track text lines it should degrade like ToMe (slowly on DocVQA); if they do not it
collapses like fixed pooling. Strongest counter-argument: the 2x DRIP number may owe most of its OCR score to the
50% of tokens kept almost untouched, an effect that disappears at 8x.

## Hypothesis backlog
1. **Smokes (cheapest, discriminates whether the launcher works).** One 15-min `debug-nextgen` training smoke per
   method (`DATASETS=llava_665k%1`, `SUFFIX=_smoke`) for fixed 4x, random 4x, DRIP 4x. Prediction: all three
   reach a finite loss below 1.3 by step 5 and write `checkpoint-*` plus (DRIP) `drip.bin` (confidence 0.85;
   the only untested path is `vtc_method=fixed|random` through Jack's launcher, which Yusen's `SFT.sh` has run).
2. **Train all six models** (fixed/random/DRIP x 4x/8x) with the 2x recipe, then evaluate each on both suites.
   Predictions (confidence 0.6): fixed+SFT 4x MMBench 41.2 -> 58 +/- 5, DocVQA 10.6 -> 18 +/- 5;
   random+SFT 4x MMBench 61.0 -> 67 +/- 3, DocVQA 14.4 -> 22 +/- 5; DRIP 4x MMBench 70 +/- 3, DocVQA 58 +/- 8
   (above fixed/random, below ToMe 67.0). 8x: fixed+SFT MMBench 29.6 -> 45 +/- 6; random+SFT 41.7 -> 55 +/- 5;
   DRIP MMBench 60 +/- 5, DocVQA 32 +/- 8 (vs ToMe 38.5).
3. **Only if a DRIP 4x/8x row is below random pruning + SFT on DocVQA:** one extra DRIP run at that ratio with
   `--drip_temperature 0.5` (the 2x temperature sweep was flat, so this applies only on a surprise).
   Prediction: within 2 pts of the 0.1 run (confidence 0.6).
4. **Only if budget remains after 1-3:** NEO-2B-SFT (LLaVA@66K) uncompressed-base rows need nothing; instead
   evaluate the DRIP 4x/8x checkpoints with naive top-k (comment the two-tailed block out in a copy of the
   compressor, selectable by a flag) to report the two-tailed delta at higher ratios. Prediction: two-tailed
   beats naive by more than 5 DocVQA at 4x (confidence 0.5).

## Constraints
- Files the agent may change: a new `VLMEvalKit/vlmeval/VLMTrainKit/SFT_vtc.sh` (generalise `SFT_perceiver.sh`:
  `VTC_METHOD=fixed|random|drip`, `RATIO`, `DATASETS`, `SUFFIX`, `COMP_LR`; output dir
  `$SCRATCH/VTC-Native-Eval-ckpts/output/NEO_continue_SFT_LLaVA_OV_llava66k_{Fixed,Random,DRIP}_{4,8}x`);
  new keys in `VLMEvalKit/vlmeval/config.py` appended in ONE new block at the end of `neo_series`
  (`NEO-2B-SFT-OV-LLaVA-llava66k-{Fixed,Random,DRIP}-{4,8}x`, mirror the existing 2x keys but point
  `lora_path`/`drip_path` at the new dirs); `scripts/eval_table2.sbatch` and `SFT_vtc.sh` may read
  `REPO=${REPO:-$SCRATCH/VTC-Native-Eval}` so they run from this direction's checkout; a small
  `scripts/summarize_models.py` (or `--models` flag on `summarize_table2.py`) that prints the 8 scores for
  arbitrary config keys; `research/` records; `vlmeval/results/training_based.csv` rows for the new cells.
- Files that may NOT change: anything under `compressors/` (DRIP two-tailed 5:5 is hardcoded in
  `drip_modules/BP.py:139` and is the protocol), `modeling_*.py`, `neo_chat.py`, `trainer.py`
  (`DRIP_LR = 1e-3`), `build.py`, `argument.py`, the existing config keys, `summarize_table2.py` row logic,
  the tex `VTC-Native-Eval.tex` at the repo root (read it for the table layout and the 2x rows; report new rows in the record, the user pastes them).
- Fixed protocol (identical to the 2x rows): base `Paranioar/NEO1_0-2B-SFT`; data `llava_665k%10`
  (`/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_665K`, read-only, already used by Jack's perceiver runs);
  LoRA lr 2e-5, global batch 128 (1 GPU x grad_accum 128), 1 epoch = 488 steps, warmup 10, max_grad_norm 1,
  `--max_pixels 4194304 --min_pixels 1310720`, DRIP `--drip_temperature 0.1`; checkpoint = final (`checkpoint-488`,
  weights saved at the output dir root), never an intermediate one. Eval: `scripts/eval_table2.sbatch` with
  `MODELS="<keys>"`, both suites (general: MMBench_DEV_EN, MME, MMMU_DEV_VAL, RealWorldQA; ocr: TextVQA_VAL,
  DocVQA_VAL, OCRBench, ChartQA_TEST), `HF_HUB_OFFLINE=1`, metric conventions of `summarize_table2.py`.
- Per-experiment ceiling: one training = 1 A100-40GB, measured 1.4 min/step x 488 = 11.4 h. Training jobs are chains of 4 h `nextgen` segments: `--time=4:00:00`, each segment resubmitted with `--dependency=afterany:<prev>` (3 segments cover the measured 11.4 h; the trainer resumes from the latest `checkpoint-*`, saved every 10 steps = 14 min, so at most 14 min is lost per boundary). Submit the whole chain at once and record every segment id.
  One model's eval =
  8 array tasks, 2B walltimes 23/13/13/6/46/114/26/27 min -> about 4.5 GPU-h. Whole direction: 6 x 11.4 + 6 x 4.5
  = 95 GPU-h. Do not start 9B rows.
- OSC: run from this direction's checkout `/fs/scratch/PAS2836/jacktaylor/wt/vtc/sft-4x8x` (env from
  `source activate $SCRATCH/conda/envs/neo`, `HF_HOME=$SCRATCH/hf`, `LMUData=$SCRATCH/LMUData`). Outputs land in
  that checkout's `VLMEvalKit/outputs/`; checkpoints in the shared `$SCRATCH/VTC-Native-Eval-ckpts/output/`.
  Record the commit SHA with every row (Yusen needs it).
- Every record: prediction with confidence before submission (which cells, by how much), score afterwards, and
  a PNG of score vs compression ratio (2x/4x/8x; MMBench and DocVQA panels; fixed/random/DRIP + SFT vs the
  training-free fixed/random/ToMe rows from `vlmeval/results/training_free_2B.csv`).

## Log
- 2026-11-07 wake 1: wrote SFT_vtc.sh + 4x/8x config keys (9a659cf); cloned OSC checkout; e001 smokes (fixed/random/DRIP 4x, DRIP resume test) on debug-nextgen: 8388118-8388121.
- 2026-11-09 wake 2: e001 fixed/random 4x smokes pass (87-88 s/step, files written); DRIP crashed step 1 on torch Binomial validation of the soft boundary count (ratio-independent, so the 2x run had validation off); fixed in SFT_vtc.sh (drip-only wrapper, 3b0d626). Submitted e002 DRIP 4x re-smoke A/B (8453771-8453772, debug-nextgen) and e003 fixed+random 4x training chains, 4x4 h each (8453773-8453780, ~25 GPU-h).
- 2026-11-09 wake 3: e002 DRIP fresh start OK (94.6 s/step, drip.bin saved) but resume crashed (train.py `logging.info`); e004 found a 2nd resume bug (stock loader wants full weights, PeftModel nested); both fixed in the SFT_vtc.sh wrapper (063a171, trainer.py untouched, monkeypatched `_load_from_checkpoint` = perceiver-learned 43311d8 logic), resume verified for DRIP and fixed (8454476/8454477). Cancelled e003 segments 2-4 (old launcher spooled, would crash) and resubmitted 8454543-8454548 on running seg 1; submitted e005 DRIP 4x chain 8454549-8454552 (~13.5 GPU-h). 12 in flight.
- 2026-11-09 wake 4: e003 segment 1 (8453773/8453777) TIMEOUT as planned at step 166/164; segment 2 (8454543/8454546) resumed from checkpoint-160 cleanly (480 LoRA tensors). e005 DRIP seg 1 running (3 h). Submitted e006 8x launcher smokes, fixed + DRIP, debug-nextgen (8462191-8462192). 8x chains wait for e006 and for free slots (cap 12). Committed GPU-h estimate: ~107 of 120.
- 2026-11-09 wake 5: e006 8x smokes pass (fixed 90.7 s/step, DRIP 92.0; DRIP boundary_loss 0.063 and pre-clip grad_norm 12 vs 0.025/2.0 at 4x, watch). Submitted e007 DRIP 8x chain segments 1-2 (8463470-8463471, nextgen, ad84457); segments 3-4 and fixed/random 8x chains wait for slots (12/12 in flight). 4x chains: fixed/random at seg 2, DRIP at seg 1 (step ~140).
- 2026-11-09 wake 6: e007 DRIP 8x segments 1-2 both FAILED in 1-2 min on node a0233 (CUDA device busy/unavailable at set_device, 0 steps, 0.06 GPU-h): node fault, not code. Retried as e008 with `--exclude=a0233`: 8463947-8463948 (e7e47e4). Segments 3-4 and fixed/random 8x chains still wait for slots (12/12). A second identical CUDA failure on another node = pause.
- 2026-11-10 wake 7: e005 DRIP 4x seg 1 TIMEOUT as planned at checkpoint-160 (88.5 s/step, 4.0 GPU-h); segs 2-3 (8454550/8454551) both FAILED in 1.5 min on node a0233, same CUDA set_device fault as e007 (node, not code). Excluded a0233 on all pending segments via scontrol, appended DRIP 4x segs 5-6 (8464684-8464685) and DRIP 8x seg 3 (8464719), all `--exclude=a0233`. spent_gpu_hours recomputed from sacct over finished jobs only (12.86; the earlier 15.86 counted running time). 12/12 in flight. Worth telling OSC that a0233 refuses CUDA contexts.
- 2026-11-10 wake 8: e003 seg 2 (8454543/8454546) TIMEOUT as planned at step 326, checkpoint-320, 84.5 s/step; seg 3 resumed cleanly (480 LoRA tensors), seg 4 will finish the last ~8 steps. Submitted e009 fixed + random 8x segment 1 (8470989-8470990, nextgen, exclude a0233, ee905ad); segments 2-4 and DRIP 8x seg 4 wait for slots (12/12). spent_gpu_hours = 20.86 from sacct over finished jobs (26.01 from the driver counted running time). Projection ~104 of 120 GPU-h.
- 2026-11-10 wake 9: e005 DRIP 4x seg 4 (8454552) TIMEOUT as planned at checkpoint-320 (85.5 s/step), seg 5 resumed; e008 DRIP 8x seg 1 (8463947, node a0341, so the a0233 exclusion works) TIMEOUT as planned at checkpoint-160 (88.7 s/step), seg 2 resumed. Submitted e009 fixed/random 8x segment 2 (8474120-8474121, afterany, ecf8110). 12/12 in flight; still to append: e009 segs 3-4 (x2), DRIP 8x seg 4. spent_gpu_hours 28.87 from finished jobs (driver's 35.94 counted running time). Evals start as each chain ends.
- 2026-11-10 wake 10: e003 DONE: fixed + random 4x reached step 488 (root adapters, last-20 loss 0.94 / 0.91 vs 1.20 / 1.08 first-20; 24.83 GPU-h). DRIP 4x seg 5 and DRIP 8x seg 2, fixed/random 8x seg 2 TIMEOUT as planned (ckpt 470 / 310 / 310 / 320). DRIP 4x seg 6 and DRIP 8x seg 3 FAILED at CUDA set_device on node a0362 (same fault as a0233, which OSC drained for reboot today): node, not code, so retried with `--exclude=a0233,a0362` instead of pausing. Submitted (8779b0d): DRIP 4x seg 7 on debug-nextgen (8489186, 18 steps), DRIP 8x segs 4-5 (8489189-90), fixed 8x segs 3-4 (8489191-92), random 8x segs 3-4 (8489193, 8489195); e010 4x evals: debug-nextgen smoke Fixed-4x/RealWorldQA 8489180, then Fixed+Random general/ocr arrays 8489181/8489185 and DRIP-4x 8489187/8489188 (afterok on smoke and seg 7). 12/12 in flight. spent 61.77 (sacct, finished jobs); projection +~27 this round +~13.5 for 8x evals = ~102 of 120.
- 2026-11-10 wake 11: e005 DONE: DRIP 4x seg 7 (8489186, debug-nextgen) reached step 488, root adapter + drip.bin (last-20 loss 0.928, boundary_loss max 0.026; 12.57 GPU-h, 3 segments lost to nodes). e010 smoke passed (Fixed-4x RWQA 51.50); 9/22 4x eval tasks in: fixed+SFT 4x MME 1349 (tf 781), random 1428 (tf 1210), OCRBench 3.0 / 5.6 (tf 2.1 / 4.6), so SFT recovers general VQA, not OCR; MMBench/DocVQA cells still running. Submitted e011 8x evals for Fixed/Random/DRIP (8490212 general, 8490213 ocr, arrays 0-11, de28b43), afterok on the three final 8x segments. 12/12 in flight. spent 64.35 (sacct, finished); projection ~64 + 12 running 8x segs + ~6 e010 rest + ~9 e011 = ~91 of 120.
- 2026-11-10 wake 12: e010 general suites done (8489181, 8489187 archived, all rc=0). 4x MMBench: fixed+SFT 64.26 (tf 41.24, above the 58 +/- 5 band), random+SFT 67.78 (hit), DRIP 68.47 (hit; ToMe tf 69.67). OCR stays flat for fixed/random (ChartQA 21.3 vs tf 19.1) and DRIP 4x OCRBench fell to 11.6 (2x 40.6, ToMe 4x 16.6): early sign DRIP's OCR edge shrinks at 4x. DocVQA cells still running (8489185, 8489188). Nothing new submitted: backlog item 3 needs the DRIP vs random DocVQA cells; 8x chains at segs 3-4, e011 evals gated afterok. 10 ids in flight. spent 67.59 (sacct, finished; driver's 74.13 counted running time).
