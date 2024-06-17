#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace cuda_kernels::kernels {

__global__ void flash_attn_fwd_f32_kernel(
    const float* Q, const float* K, const float* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    const float softmax_scale, float* l, float* m, float* O);

void flash_attn_fwd(const torch::Tensor& Q, const torch::Tensor& K,
                             const torch::Tensor& V, torch::Tensor& O);

}  // namespace cuda_kernels::kernels