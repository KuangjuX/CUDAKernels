import torch
torch.ops.load_library("build/src/libcuda_kernels.so")


def reduce_sum(input, output, size):
    return torch.ops.cuda_kernels.reduce_sum(input, output, size)
