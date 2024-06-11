#pragma once

#include "cuda_utils.hpp"

#include <torch/script.h>

namespace cuda_kernels::kernels {

template <typename Element, const int THREAD_NUMS, const int WARP_SIZE>
__global__ void softmax_kernel(const Element* x, Element* y, const int size);

// Safe softmax kernel that avoids overflow and underflow
// safe_softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
template <typename Element, const int THREAD_NUMS, const int WARP_SIZE>
__global__ void safe_softmax_kernel(const Element* x, Element* y,
                                    const int size);

void softmax(const torch::Tensor& input, torch::Tensor& output, int64_t size);

}  // namespace cuda_kernels::kernels