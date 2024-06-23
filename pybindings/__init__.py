import torch
torch.ops.load_library("build/src/libcuda_kernels.so")


def reduce_sum(input, output, size):
    return torch.ops.cuda_kernels.reduce_sum(input, output, size)


def reduce_max(input, output, size):
    return torch.ops.cuda_kernels.reduce_max(input, output, size)


def softmax(input, output, size):
    return torch.ops.cuda_kernels.softmax(input, output, size)


def flash_attn_fwd(q, k, v, o):
    return torch.ops.cuda_kernels.flash_attn_fwd(q, k, v, o)


def vec_copy_tensor_g2r(input, output, size):
    return torch.ops.cuda_kernels.vec_copy_g2r(input, output, size)


def copy_2d_tensor_g2r(input, output, height, width):
    return torch.ops.cuda_kernels.copy_2d_tile_g2r(input, output, height, width)
