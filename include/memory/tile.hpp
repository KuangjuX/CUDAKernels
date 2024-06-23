#pragma once

#include "common.hpp"
#include "cuda_utils.hpp"
#include "memory/types/register.hpp"

namespace cuda_kernels::memory {

// refs:
// https://github.com/HazyResearch/ThunderKittens/blob/main/src/ops/group/memory/tile/global_to_register.cuh
/**
 * @brief Copy a row-major 2D tile from global memory to register tile
 * @param src[in] source data in global memory
 * @param dst[out] destination data in register tile
 * @param row_stride[in] row stride of source data
 */
template <typename T, typename T2, size_t height, size_t width>
__global__ void copy_2d_tile_g2r(const T* src,
                                 types::RegTile<T2, height, width>& dst,
                                 const int row_stride) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int row_offset = dst.rows * warp_id;

    // 这里加载的 layout 完全按照 ldmatrix 来加载，对于一个 16 * 16 的
    // subtile，按照 `ldmatrix` 的方式将其 分成 4 个 8 * 8 的矩阵，然后一个 warp
    // 中的每个线程从 4 个 8 * 8 的矩阵中向量化地加载 2 个元素（可能是 2/4
    // 个寄存器） 到线程私有寄存器中，这里的格式是严格按照 tensor core 加载的，
    // 在计算矩阵乘时 mma 指令会自动 warp shuffle 线程私有寄存器的值。
    // References:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-load-instruction-ldmatrix
#pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        int row = row_offset + i * dst.tile_size + (lane_id / 4);
#pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            int col = j * dst.tile_size + (lane_id % 4);
            dst.tiles[i][j].data[0].x = src[(row + 0) * row_stride + col + 0];
            dst.tiles[i][j].data[0].y = src[(row + 0) * row_stride + col + 1];
            dst.tiles[i][j].data[2].x = src[(row + 0) * row_stride + col + 8];
            dst.tiles[i][j].data[2].y = src[(row + 0) * row_stride + col + 9];
        }

#pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            int col = j * dst.tile_size + (lane_id % 4);
            dst.tiles[i][j].data[1].x = src[(row + 8) * row_stride + col + 0];
            dst.tiles[i][j].data[1].y = src[(row + 8) * row_stride + col + 1];
            dst.tiles[i][j].data[3].x = src[(row + 8) * row_stride + col + 8];
            dst.tiles[i][j].data[3].y = src[(row + 8) * row_stride + col + 9];
        }
    }
}

}  // namespace cuda_kernels::memory