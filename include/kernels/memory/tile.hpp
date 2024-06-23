#pragma once

#include "cuda_utils.hpp"

namespace cuda_kernels::kernels {

template <typename Element>
__global__ void copy_2d_tile_g2r_kernel(const Element* src, Element* dst);

}