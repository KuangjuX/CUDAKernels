#pragma once

#include "cuda_utils.hpp"

namespace cuda_kernels::kernels {

DEVICE void flash_attn_fwd_f32_kernel(const float* Q, const float* K,
                                      const float* V, const int N, const int d,
                                      const int Tc, const int Tr, const int Bc,
                                      const int Br, const float softmax_scale,
                                      float* l, float* m, float* O);
}  // namespace cuda_kernels::kernels