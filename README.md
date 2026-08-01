# VTC-Native-Eval: Towards a Structured Study of Visual Token Compression Evaluation for Native VLMs

## Environment Setup

```bash
module load miniconda3/24.1.2-py310
conda create -n neo pip python=3.10
conda activate neo
cd VLMEvalKit
python -m pip install -e .
python -m pip install transformers==4.57.1
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
cd VLMEvalKit
python quick_demo.py
```

Expect the following response for the first question:

```
Based on the image provided, here is a thorough description:

The image is a **cartoon-style illustration** of a **tree**.

Here's a breakdown of its contents:

1.  **The Tree:**
    *   **Trunk:** The tree has a thick, sturdy trunk that is brown. It has a distinct, somewhat irregular shape, tapering slightly towards the top. The trunk is depicted with visible lines suggesting texture or bark.
    *   **Branches:** The trunk splits into several main branches that extend upwards and outwards. These branches are also brown and have a similar textured appearance to the trunk. They spread out to support the foliage.
    *   **Foliage:** The tree has a large, rounded canopy of green leaves. The leaves are depicted with a slightly wavy or scalloped edge, giving them a somewhat stylized look. The green color is a uniform shade of green.

2.  **The Apples:**
    *   **Quantity:** There are a total of **eight red apples** visible on the tree.
    *   **Location:** The apples are scattered across the tree's canopy, attached to the branches. They are positioned at various heights and angles, appearing to be growing naturally.
    *   **Appearance:** Each apple is depicted as a simple, round shape with a distinct red color. It has a small, simple brown stem extending upwards from the top of the apple.

3.  **Style and Composition:**
    *   The image is a **flat, two-dimensional illustration** with bold outlines.
    *   The colors are bright and solid, typical of cartoon or educational graphics.
    *   The background is plain white, which makes the tree and apples stand out clearly.

In summary, the image is a simple, colorful drawing of a tree with a green canopy and brown trunk, bearing eight red apples.
```

and for the second question:

```
Based on the provided images:

1.  **Image 1:** This image shows a tree with 8 red apples.
2.  **Image 2:** This image shows a tree with 5 red apples.

To find the total number of apples, we add the apples from both images:

Total apples = Apples in Image 1 + Apples in Image 2
Total apples = 8 + 5
Total apples = 13

Therefore, there are a total of **13** apples in the provided images.
```


## Evaluation Benchmarks

TBD


