#pragma once

#include "cuda_utils.hpp"

namespace cudakernels::kernels {

template <typename Element>
__global__ void gemv_kernel(const Element* A, const Element* x, Element* y,
                            int m, int k);
}