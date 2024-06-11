#include "cuda_utils.hpp"

#include <type_traits>

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
        if (std::is_integral_v<Element>) {
            value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
        } else if (std::is_floating_point_v<Element>) {
            value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
        }
    }
    return value;
}

}  // namespace cuda_kernels::warp