#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace cudakernels::kernels {
template <typename Element>
__global__ void reduce_sum_kernel(const Element* input, Element* output,
                                  int size, int thread_size);

template <typename Element>
__global__ void reduce_max_kernel(const Element* input, Element* output,
                                  int size, int thread_size);

void reduce_sum(const torch::Tensor& input, torch::Tensor& output,
                int64_t size);

void reduce_max(const torch::Tensor& input, torch::Tensor& output,
                int64_t size);

}  // namespace cudakernels::kernels