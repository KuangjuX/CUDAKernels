#include "kernels/memory/tile.hpp"
#include "memory/mod.hpp"

namespace cuda_kernels::kernels {
template <typename T, typename T2>
__global__ void copy_2d_tile_g2r_kernel(const T* src, T* dst) {
    int tid = threadIdx.x;

    const int row_stride = 16;
    memory::types::RegTile<T2, 1, 1> reg;

    memory::copy_2d_tile_g2r(src, reg, row_stride);

    // Debug
    if (tid == 0) {
        for (int i = 0; i < 4; ++i) {
            printf("reg.data[%d] x = %f, y = %f\n", i,
                   reg.tiles[0][0].data[i].x, reg.tiles[0][0].data[i].y);
        }
    }
}

void copy_2d_tile_g2r(const torch::Tensor& input, torch::Tensor& output,
                      int64_t height, int64_t width) {
    dim3 block_size(32);
    dim3 grid_size(1);

    if (input.dtype() == torch::kFloat32) {
        copy_2d_tile_g2r_kernel<float, float2><<<grid_size, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>());
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}
}  // namespace cuda_kernels::kernels