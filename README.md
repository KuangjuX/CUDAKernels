# CUDAKernels

## Introduction

The Project is used to collect the CUDA kernels that I have written by hand, for the purpose of learning various CUDA techniques and conducting performance evaluation.

## Kernels

- [Reduce Sum](src/kernels/reduce.cu): 使用 Warp Reduce 实现的 reduce sum 操作 / Reduce Sum Operation with **Warp Reduce**.
- [Reduce Max](src/kernels/reduce.cu): 使用 Warp Reduce 实现的 reduce max 操作 / Reduce Max Operation with **Warp Reduce**.
- [Softmax](src/kernels/softmax.cu): 使用 Warp Reduce 实现的未分块的 softmax 操作 / Softmax Operation with **Warp Reduce**.

## Notes

- [Vectorized Memory Access](notes/memory/vec.md): 向量化内存访问笔记 / Notes about Vectorized Memory Access.
- [Memory Coalescing](notes/memory/coalescing.md): 内存合并访问笔记 / Notes about Memory Coalescing.
- [Warp-Level Primitives](notes/warp.md): Warp 原语笔记 / Notes about Warp-Level Primitives.
- [FlashAttention](notes/flash_attn.md): FlashAttention 笔记 / Notes about FlashAttention.

## References

- [CUDA-Learn-Notes: 🎉CUDA 笔记 / 大模型手撕CUDA / C++笔记，更新随缘: flash_attn、sgemm、sgemv、warp reduce、block reduce、dot product、elementwise、softmax、layernorm、rmsnorm、hist etc.](https://github.com/DefTruth/CUDA-Learn-Notes)
- [ThunderKittens: Tile primitives for speedy kernels](https://github.com/HazyResearch/ThunderKittens)