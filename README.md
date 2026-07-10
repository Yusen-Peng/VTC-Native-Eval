# VTC-Native-Eval: Towards a Structured Study of Visual Token Compression Evaluation for Native VLMs

## Environment

```bash
module load miniconda3/24.1.2-py310
conda create -n neo python=3.12 -y
conda activate neo
python -m pip install -r requirements.txt
```

## Neo model checkpoints

pretraining checkpoint: [Paranioar/NEO1_0-2B-PT](https://huggingface.co/Paranioar/NEO1_0-2B-PT) for 2B, [Paranioar/NEO1_0-9B-PT](https://huggingface.co/Paranioar/NEO1_0-9B-PT) for 9B

mid-training checkpoint: [Paranioar/NEO1_0-2B-MT](https://huggingface.co/Paranioar/NEO1_0-2B-MT) for 2B, [Paranioar/NEO1_0-9B-MT](https://huggingface.co/Paranioar/NEO1_0-9B-MT) for 9B

SFT checkpoint: [Paranioar/NEO1_0-2B-SFT](https://huggingface.co/Paranioar/NEO1_0-2B-SFT) for 2B, [Paranioar/NEO1_0-9B-SFT](https://huggingface.co/Paranioar/NEO1_0-9B-SFT) for 9B.


## Mid-training and SFT datasets

The NEO author's suggestion: *"We do not currently have an open-source plan. Most of the data is open source. We recommend using the mid-training and SFT data from LLaVA-OneVision-1.5, which our lab has recently open-sourced."*

Mid-training data from LLaVA-OV-1.5: [mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)

SFT data from LLaVA-OV-1.5: [mvp-lab/LLaVA-OneVision-1.5-Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)


## Quick Demo


```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:20:00
module load miniconda3/24.1.2-py310
conda activate neo
python VLMEvalKit/quick_demo.py
```

## Evaluation Benchmarks

TBD


