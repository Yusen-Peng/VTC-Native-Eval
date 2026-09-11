# VTCNativeEval: A Structured Study of Visual Token Compression Methods for Native VLMs

## Environment Setup

We closely follow the setup from [NEO repository](https://github.com/EvolvingLMMs-Lab/NEO/blob/main/VLMEvalKit/docs/en/Quickstart.md) to set up the environment:

```bash
module load miniconda3/24.1.2-py310
conda create -n neo pip python=3.10
conda activate neo
cd VLMEvalKit
python -m pip install -e .
python -m pip install transformers==4.57.1
```

then you are all set!

## Quick inference demo

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:10:00
module load miniconda3/24.1.2-py310
conda activate neo
cd VLMEvalKit
python quick_demo.py
```

## VLM Evaluation

Let's find all available datasets first:

```bash
python -c "
from vlmeval.dataset import SUPPORTED_DATASETS
for name in sorted(set(SUPPORTED_DATASETS)):
    print(name)
" > supported_datasets.txt
```

Then we can run one-liner evaluation:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:10:00
module load miniconda3/24.1.2-py310
conda activate neo
cd VLMEvalKit
export LMUData=/fs/scratch/PAS2836/yusenpeng_dataset/VLMEvalKit_data
# MMBench
python run.py --data MMBench_DEV_EN --model NEO-2B-SFT --verbose
# MME
python run.py --data MME --model NEO-2B-SFT --verbose
# MMMU
python run.py --data MMMU_DEV_VAL --model NEO-2B-SFT --verbose
# RealWorldQA
python run.py --data RealWorldQA --model NEO-2B-SFT --verbose
# TextVQA
python run.py --data TextVQA_VAL --model NEO-2B-SFT --verbose
# DocVQA
python run.py --data DocVQA_VAL --model NEO-2B-SFT --verbose
# OCRBench
python run.py --data OCRBench --model NEO-2B-SFT --verbose
# ChartQA
python run.py --data ChartQA_TEST --model NEO-2B-SFT --verbose
```

We update the results on Overleaf: [VTC-Native-Eval](https://www.overleaf.com/read/wjvqxyygyjpn#73895d)

## Neo model checkpoints

pretraining checkpoint: [Paranioar/NEO1_0-2B-PT](https://huggingface.co/Paranioar/NEO1_0-2B-PT) for 2B, [Paranioar/NEO1_0-9B-PT](https://huggingface.co/Paranioar/NEO1_0-9B-PT) for 9B

mid-training checkpoint: [Paranioar/NEO1_0-2B-MT](https://huggingface.co/Paranioar/NEO1_0-2B-MT) for 2B, [Paranioar/NEO1_0-9B-MT](https://huggingface.co/Paranioar/NEO1_0-9B-MT) for 9B

SFT checkpoint: [Paranioar/NEO1_0-2B-SFT](https://huggingface.co/Paranioar/NEO1_0-2B-SFT) for 2B, [Paranioar/NEO1_0-9B-SFT](https://huggingface.co/Paranioar/NEO1_0-9B-SFT) for 9B.


## Mid-training and SFT datasets

The NEO author's suggestion: *"We do not currently have an open-source plan. Most of the data is open source. We recommend using the mid-training and SFT data from LLaVA-OneVision-1.5, which our lab has recently open-sourced."*

Mid-training data from LLaVA-OV-1.5: [mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)

SFT data from LLaVA-OV-1.5: [mvp-lab/LLaVA-OneVision-1.5-Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)



