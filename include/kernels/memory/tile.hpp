#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace cuda_kernels::kernels {

template <typename T, typename T2>

__global__ void copy_2d_tile_g2r_kernel(const T* src, T* dst);

void copy_2d_tile_g2r(const torch::Tensor& input, torch::Tensor& output,
                      int64_t height, int64_t width);

}  // namespace cuda_kernels::kernels