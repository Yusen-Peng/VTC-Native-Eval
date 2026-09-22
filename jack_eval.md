# NEO-2B-SFT evaluation (jacktaylor run, 2026-09-06)

Full-resolution NEO-2B-SFT (`Paranioar/NEO1_0-2B-SFT`, snapshot `3f480ac`), no visual token
compression. VLMEvalKit 1.0 (`VLMEvalKit/`), repo commit `2b89d1e`, transformers 4.57.1,
torch 2.14.0+cu130. Each benchmark ran as its own Slurm array task on OSC Ascend
(`nextgen`, 1x A100-40GB, 4 h walltime) via `eval_neo2b.sbatch`; jobs 7174117 / 7174155.
Raw outputs: `/fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval/VLMEvalKit/outputs/NEO-2B-SFT/`.

## Summary

| Benchmark | Split | Score | Metric | Wall time | README ref |
| --------- | ----- | ----- | ------ | --------- | ---------- |
| MMBench | DEV_EN | **76.20** | acc | 23 min | 76.20 |
| MME | - | **1565.7** | perception (reasoning 532.9) | 13 min | 1565.7 |
| MMMU | DEV_VAL (val) | **48.33** | acc (dev 52.0) | 13 min | 48.33 |
| GQA | TestDev_Balanced | **53.99** | acc | 1 h 28 min | - |
| RealWorldQA | - | **62.75** | acc | 6 min | - |
| TextVQA | VAL | **73.94** | acc | 46 min | 73.94 |
| DocVQA | VAL | **89.86** | ANLS | 1 h 54 min | 89.86 |
| OCRBench | - | **77.0** | final score 770/1000 | 26 min | 77.0 |
| OCRBench v2 | - | incomplete | see below | >4 h (TIMEOUT) | running |
| ChartQA | TEST | **81.0** | relaxed acc (aug 94.32 / human 67.68) | 27 min | wait |

All previously reported numbers in `README.md` reproduce exactly (MMBench, MME, MMMU, TextVQA,
DocVQA, OCRBench). New numbers: GQA 53.99, RealWorldQA 62.75, ChartQA 81.0.

## Per-benchmark breakdown

MMBench_DEV_EN (overall 76.20): AR 81.91, CP 86.49, FP-C 66.43, FP-S 74.40, LR 55.93, RR 77.39.
Weakest sub-tasks: spatial_relationship 33.3, future_prediction 37.5, object_localization 54.3.

MME: perception 1565.70, reasoning 532.86. OCR 200.0, existence 195.0, text_translation 192.5,
color 165.0, landmark 159.0, scene 157.5, celebrity 150.6, position 145.0, artwork 135.75,
code_reasoning 132.5, count 131.7, posters 126.2, commonsense_reasoning 122.9,
numerical_calculation 85.0.

MMMU_DEV_VAL: validation 48.33 (Art & Design 61.67, Business 40.0, Health & Medicine 50.0,
Humanities & Social Science 65.83, Science 34.0, Tech & Engineering 45.71); dev 52.0.

GQA_TestDev_Balanced (overall 53.99): choose 81.58, compare 58.23, logical 71.10, query 36.94,
verify 76.87.

ChartQA_TEST (overall 81.0): test_augmented 94.32, test_human 67.68.

OCRBench (770): Text Recognition 268, Scene Text-centric VQA 160, Doc-oriented VQA 148,
Key Information Extraction 122, Handwritten Math Expression Recognition 72.

## OCRBench v2: not finished

Task 7174117_9 hit the 4 h walltime at 1918/10000 samples. Throughput collapsed from ~1.9 it/s
(first ~400 samples) to ~14 s/it; the model greedy-decodes up to `max_new_tokens=4096`
(`vlmeval/vlm/neo/neo_chat.py:248`) and some samples degenerate into repetition (longest
prediction 13,125 chars, a repeating JSON list). At the observed rate the full 10k set needs
roughly 30-40 GPU-hours on one A100-40GB.

VLMEvalKit checkpointed 1910 predictions in
`outputs/NEO-2B-SFT/T20260906_G2b89d1e4/01_OCRBench_v2.pkl`; resubmitting
`sbatch --array=9 --time=<longer> eval_neo2b.sbatch` resumes from there. Options to make it
tractable: lower `max_new_tokens` for OCRBench_v2 (e.g. 1024), add a repetition penalty, or split
the TSV across several array tasks.
