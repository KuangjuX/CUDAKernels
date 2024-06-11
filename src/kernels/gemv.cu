#include "kernels/mod.hpp"

namespace cuda_kernels::kernels {
__global__ void gemv_kernel(const float* A, const float* x, float* y, int m,
                            int n) {}
}  // namespace cudakernels::kernels