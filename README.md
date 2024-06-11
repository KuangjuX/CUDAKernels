# CUDAKernels

## Introduction

The Project is used to collect the CUDA kernels that I have written by hand, for the purpose of learning various CUDA techniques and conducting performance evaluation.

## Kernels
- [Reduce Sum](src/kernels/reduce.cu): 使用 Warp Reduce 实现的 reduce sum 操作。
- [Reduce Max](src/kernels/reduce.cu): 使用 Warp Reduce 实现的 reduce max 操作。

## References
- [CUDA-Learn-Notes: 🎉CUDA 笔记 / 大模型手撕CUDA / C++笔记，更新随缘: flash_attn、sgemm、sgemv、warp reduce、block reduce、dot product、elementwise、softmax、layernorm、rmsnorm、hist etc.](https://github.com/DefTruth/CUDA-Learn-Notes)
- [ThunderKittens: Tile primitives for speedy kernels](https://github.com/HazyResearch/ThunderKittens)