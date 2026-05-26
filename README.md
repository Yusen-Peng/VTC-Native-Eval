# VTC-Eval: comprehensive and structured Visual Token Compression Evaluation

## Research Questions

We evaluate a collection of **10** current visual token compression methods by asking the following **5** research questions:

1. Does **late compression** within LLM decoder always perform better than **early compression** after image encoder?
2. Do **simple baselines** such as fixed pooling and random pruning outperform the state-of-the-art?
4. Do **text-guided** methods always outperform **text-agnostic** methods?
3. Does **finetuning** bring performance recovery over the popular training-free approaches?
5. Which VQA benchmarks are "easy" (insensitive to compression rate) and which are "hard"?

## Method Collection

| method name | baseline? | compression stage? | text guidance? |
| ----------- | --------- | ------------------ | -------------- |
| fixed pooling | yes | early | no |
| random pruning | yes | early | no |
| LLaVA-PruMerge | no | early | no |
| LLaVA-PruMerge++ | no | early | no |
| ToME | no | early | no |
| FastV | no | early | no |
| G-Prune | no | early | no |
| VTW | no | late | no |
| SparseVLM | no | hybrid | yes |
| VisionTrim | no | hybrid | yes |


## Evaluation Benchmarks


