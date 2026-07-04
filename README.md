# VTC-Eval: Towards a Structured Study of Visual Token Compression Evaluation

## Current benchmark/evaluation

[Arxiv 2025] [UniPruneBench](https://arxiv.org/abs/2511.02650):

1. underrated baseline: *"Random pruning remains a surprisingly strong baseline."*
2. no consistent winner: *"No single method achieves universal superiority."*

[ACL 2026] [VTC-Bench](https://aclanthology.org/2026.acl-long.195): 

1. filter "too hard" questions: *"drop out the samples answered incorrectly at the original resolution, which we consider are too hard for the original models to understand."* 
2. decouple "hard" and "easy" questions: *Difficult Samples (Group A): Samples that are answered incorrectly by the downsampling method* and vice versa.

**Limitations of current evaluation/benchmarking** - relatively comprehensive, but with limited insights!

## Research Questions

We evaluate a collection of **10** current visual token compression methods by asking the following **5** research questions:

1. **Early compression v.s. Late compression**: Does late compression within LLM decoder always perform better than early compression after image encoder?
2. **Text-guided v.s. Text-agnostic**: Do text-guided methods always outperform text-agnostic methods?
3. **Lossy v.s. Lossless**: Which VQA benchmarks are "easy" (insensitive to compression rate) and which are "hard"?
4. **Training-based v.s. Training-free**: Does finetuning bring performance recovery over the popular training-free approaches?
5. **Encoder-based v.s. Encoder-free**: Do encoder-free VLMs present different trend when it comes to visual compression? 

## Method Collection

| method name | baseline? | compression stage? | text guidance? |
| ----------- | --------- | ------------------ | -------------- |
| fixed pooling | yes | early | no |
| random pruning | yes | early | no |
| LLaVA-PruMerge | no | early | no |
| PruneSID | no | early | no |
| ToME | no | early | no |
| FastV | no | early | no |
| G-Prune | no | early | no |
| VTW | no | late | no |
| SparseVLM | no | hybrid | yes |
| VisionTrim | no | hybrid | yes |


## Evaluation Benchmarks


