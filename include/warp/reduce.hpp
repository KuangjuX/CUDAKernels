#include "cuda_utils.hpp"

namespace cuda_kernels::warp {
template <typename Element, const int kWarpSize = 32>
DEVICE Element warp_reduce_sum(Element value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset >= 1; offset /= 2) {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

template <typename Element, const int kWarpSize = 32>
DEVICE Element warp_reduce_max(Element value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset >= 1; offset /= 2) {
        value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
    }
    return value;
}

}  // namespace cuda_kernels::warp