#pragma once

#include "common.hpp"
#include "cuda_utils.hpp"
#include "memory/types/register.hpp"

namespace cuda_kernels::memory {

template <typename T, typename T2, size_t height, size_t width>
__global__ void copy_2d_tile_g2r(const T* src,
                                 types::RegTile<T2, height, width>& dst,
                                 const int row_stride) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int row_offset = dst.rows * warp_id;

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