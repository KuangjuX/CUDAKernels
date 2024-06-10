#include "cuda_utils.hpp"

namespace cudakernels::warp {
template <typename Element, const int kWarpSize = 32>
DEVICE Element warp_reduce_sum(Element value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

template <typename Element, const int kWarpSize = 32>
DEVICE Element warp_reduce_max(Element value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    }
    return value;
}

}