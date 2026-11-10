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
spent_gpu_hours: 1.4
max_jobs_in_flight: 12
jobs:
- 8453771
- 8453772
- 8453773
- 8453774
- 8453775
- 8453776
- 8453777
- 8453778
- 8453779
- 8453780
jobs_archived:
- 8388118
- 8388119
- 8388120
- 8388121
last_wake: '2026-11-09T18:00:01-05:00'
next_id: 4
model: null
effort: null
wakes: 2
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
