#include "kernels/mod.hpp"

#include <torch/script.h>

namespace cuda_kernels {
TORCH_LIBRARY(cuda_kernels, c) {
    c.def("reduce_sum", &kernels::reduce_sum);
    c.def("reduce_max", &kernels::reduce_max);
    c.def("softmax", &kernels::softmax);
    c.def("flash_attn_fwd", &kernels::flash_attn_fwd);
    c.def("vec_copy_g2r", &kernels::vec_copy_g2r);
    c.def("copy_2d_tile_g2r", &kernels::copy_2d_tile_g2r);
};
}  // namespace cuda_kernels