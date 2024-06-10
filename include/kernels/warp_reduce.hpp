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

}  // namespace cudakernels::kernels