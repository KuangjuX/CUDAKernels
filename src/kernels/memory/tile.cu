#include "kernels/memory/tile.hpp"
#include "memory/mod.hpp"
#include "warp/mod.hpp"

namespace cuda_kernels::kernels {
template <typename T, typename T2>
__global__ void copy_2d_tile_g2r_kernel(const T* src, T* dst) {
    int tid = threadIdx.x;

    const int row_stride = 16;
    memory::types::RegTile<T2, 1, 1> reg0;
    memory::types::RegTile<T2, 1, 1> reg1;

    memory::copy_2d_tile_g2r(src, reg0, row_stride);
    warp::ldmatrix<T, T2>(src, reg1);

    // Debug
    if (tid == 0) {
        // for (int i = 0; i < 4; ++i) {
        //     printf("reg0.data[%d] x = %f, y = %f\n", i,
        //            __half2float(reg0.tiles[0][0].data[i].x),
        //            __half2float(reg0.tiles[0][0].data[i].y));
        // }

        for (int i = 0; i < 4; ++i) {
            printf("reg1.data[%d] x = %f, y = %f\n", i,
                   __half2float(reg1.tiles[0][0].data[i].x),
                   __half2float(reg1.tiles[0][0].data[i].y));
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
    } else if (input.dtype() == torch::kFloat16) {
        copy_2d_tile_g2r_kernel<__half, half2><<<grid_size, block_size>>>(
            reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}
}  // namespace cuda_kernels::kernels