#include "kernels/mod.hpp"

#include <torch/script.h>

namespace cuda_kernels {
TORCH_LIBRARY(cuda_kernels, c) {
    c.def("reduce_sum", &kernels::reduce_sum);
    c.def("reduce_max", &kernels::reduce_max);
};
}  // namespace cuda_kernels