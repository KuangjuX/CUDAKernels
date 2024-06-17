#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace cuda_kernels::kernels {
template <int OUTER_LENGTH>
__global__ void vec_copy_g2r_kernel_f32(const float* src, float* dst,
                                        int length);

void vec_copy_g2r(const torch::Tensor& input, torch::Tensor& output,
                  int64_t size);
}  // namespace cuda_kernels::kernels