---
slug: fastv-k16
title: "Does FastV pruning after layer 16 (PyramidDrop's layer) beat FastV-2 and match PyramidDrop-16 at equal FLOPs on NEO-2B?"
project: vtc
type: direction
cluster: ascend
status: active
wake: on_jobs
budget_gpu_hours: 30
spent_gpu_hours: 0
max_jobs_in_flight: 8
jobs: []
jobs_archived: []
last_wake: null
next_id: 1
model: null
effort: null
---
# Does FastV pruning after layer 16 beat FastV-2 and match PyramidDrop-16 at equal FLOPs?

Frontmatter fields: `status` active|paused|done. `wake` now|on_jobs. `jobs` = ids in flight; finished ids move to
`jobs_archived` (never delete ids). `budget_gpu_hours` is a hard stop.

## Goal
Add `FastV-16` rows (prune once after decoder layer K=16, keep ratio 1/2, 1/4, 1/8 of each image's tokens) to
`tab:results-2B` (2B only; the 9B rows are out of scope for now) next to the existing `FastV-2` and `PyramidDrop-16` rows, with prefill
TFLOPs (and latency if the existing latency procedure is reproducible). Existing rows, 2B / 9B, DocVQA and
MMBench, TFLOPs:

| method | 2x | 4x | 8x |
|---|---|---|---|
| FastV-2, 2B | 82.00 DocVQA / 73.80 MMB, 6.19 TF | 63.92 / 65.46, 3.23 | 39.99 / 56.62, 1.94 |
| PyramidDrop-16, 2B | 81.54 / 74.22, 8.92 | 55.87 / 69.75, 7.01 | 29.68 / 58.16, 6.24 |
| FastV-2, 9B | 65.37 / 81.01, 22.12 | 38.09 / 75.52, 11.78 | 20.82 / 68.30, 7.00 |
| PyramidDrop-16, 9B | 82.65 / 81.35, 30.47 | 66.15 / 80.75, 23.75 | 46.69 / 79.20, 20.65 |

Settled when the three 2B FastV-16 rows (3 ratios x 8 benchmarks) and their FLOPs and latency exist. The result is
positive if FastV-16 is within 1 pt of PyramidDrop-16 on DocVQA at matching TFLOPs (the 2B 2x pair is nearly
equal-cost by construction: 16 full layers of 40 plus 24 pruned layers is about 8.9 TF). It is negative if
FastV-16 stays closer to FastV-2 than to PyramidDrop-16 on OCR.

## Theory
FastV ranks image tokens by the last query token's attention after K layers. At K=2 in NEO the attention has
seen the raw patch embedding for only two layers (no CLIP encoder precedes it), so the ranking is nearly
position-driven; FastV-2 at 9B loses 23 DocVQA points at 2x while PyramidDrop-16 loses 6. PyramidDrop's own
gain over FastV in its paper is attributed partly to the later first pruning layer. If the signal is the
layer, FastV-16 should close most of the gap; if PyramidDrop's gradual multi-stage schedule matters, FastV-16
stays behind at equal cost. Counter-argument: at 8x PyramidDrop keeps all tokens through 16 layers and prunes
progressively, so FastV-16 at 8x throws away 7/8 of the tokens at once and may collapse on OCR like FastV-2 does.

## Hypothesis backlog
1. **Wire + FLOPs (no eval GPU-hours).** Add config keys `NEO-{2B,9B}-SFT-FastV-layer16-{2,4,8}x`
   (copy the `-layer2-` keys, `compression_layer=16`), add `"fastv-16": ("FastV-16", [2, 4, 8])` to `LLM_METHODS`
   in `scripts/flops.py`, run `scripts/flops.py` for 2B on `debug-nextgen` (it now also measures latency; `origin/main` commits
   00c78b2/8bf9201 added it, read the script's header), append rows to `vlmeval/results/flops.csv`. Prediction:
   2B TFLOPs 8.9 / 7.0 / 6.1 at 2x/4x/8x (within 0.2 of PyramidDrop-16's 8.92 / 7.06 / 6.24 at 2x, slightly
   below at 4x/8x), latency between FastV-2 (241 ms) and PyramidDrop-16 (367 ms) at 2x (confidence 0.8). Smoke one benchmark (RealWorldQA, 6 min) at 2B 2x on `debug-nextgen` before any array.
2. **2B, both suites, 3 ratios** (24 array tasks, about 15 GPU-h; FastV-16 keeps all tokens for 16 layers so
   expect 1.2-1.5x the FastV-2 walltimes). Predictions (confidence 0.6): 2x DocVQA 82.0 -> 84 +/- 2,
   MMBench 73.8 -> 74.5 +/- 1; 4x DocVQA 63.9 -> 68 +/- 4; 8x DocVQA 40.0 -> 44 +/- 5; ChartQA at 4x/8x stays
   above PyramidDrop-16's 37.5 / 23.5 (FastV-2 has 61.5 / 42.4).
3. **9B is out of scope for this direction** (user decision 2026-11-07); do not submit 9B jobs.
4. **Only if 2 shows FastV-16 within 2 pts of PyramidDrop-16 at 2B 2x:** K=24 at 2B 2x only (one more row,
   about 5 GPU-h) to locate where the layer effect saturates. Prediction: no further gain beyond 1 pt (0.5).

## Constraints
- Files the agent may change: new keys appended in ONE new block at the end of `neo_series` in
  `VLMEvalKit/vlmeval/config.py`; `LLM_METHODS` in `scripts/flops.py`; `vlmeval/results/flops.csv` (append rows);
  `scripts/eval_table2.sbatch` may read `REPO=${REPO:-$SCRATCH/VTC-Native-Eval}`; a small score printer for
  arbitrary config keys (`scripts/summarize_models.py` or a `--models` flag on `summarize_table2.py`);
  `research/` records and figures.
- Files that may NOT change: `compressors/llm/fastv.py`, `modeling_qwen3.py` (the K-th layer hook is
  `layer_idx + 1 == compression_layer`, already generic), `neo_chat.py`, every existing config key, the
  PyramidDrop and FastV-2 outputs, the tex.
- Fixed protocol: FastV-16 = `llm_vtc_method="fastv"`, `llm_compression_ratio` 0.5/0.25/0.125,
  `compression_layer=16`, everything else as the `-layer2-` keys; base models `Paranioar/NEO1_0-{2B,9B}-SFT`
  (40 and 42 decoder layers); eval via `scripts/eval_table2.sbatch` with `MODELS="<keys>"`, both suites,
  `HF_HUB_OFFLINE=1`, metrics as in `summarize_table2.py`; FLOPs via `scripts/flops.py` on the 2048x2048
  reference image (fvcore, prefill only). Latency: the same `scripts/flops.py` procedure that produced the tex latency column.
- Per-experiment ceiling: one array per suite, `--time` 1:30:00 (2B general), 4:00:00 (2B ocr; DocVQA 2B is
  about 2 h, resubmit with `REUSE=1` if it times out). Direction total about 15-20 GPU-h.
- OSC: run from `/fs/scratch/PAS2836/jacktaylor/wt/vtc/fastv-k16`; outputs in that checkout's
  `VLMEvalKit/outputs/`. Never rerun a finished cell (`REUSE=1`).
- Every record: prediction with confidence, score afterwards, and a PNG: DocVQA and MMBench vs TFLOPs for
  FastV-2, FastV-16, PyramidDrop-16 at 2B (Pareto view, the figure the tex already uses for
  `scaling_pareto.py`).

## Log
