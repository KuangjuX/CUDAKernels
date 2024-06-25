#pragma once

#include "cuda_utils.hpp"
#include "memory/mod.hpp"

namespace cuda_kernels::warp {

/**
 * @brief Load a matrix from shared memory to register tile with a `ldmatrix`
 * instruction.
 * @param shared_memory[in] Shared memory pointer.
 * @param reg[out] Register tile.
 * @tparam T Data type.
 * @tparam T2 Register data type.
 * @tparam height Register tile height.
 * @tparam width Register tile width.
 */
template <typename T, typename T2, const size_t height = 1,
          const size_t width = 1>
DEVICE void ldmatrix(const T* shared_memory,
                     memory::types::RegTile<T2, height, width>& reg,
                     const int row_stride) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    shared_memory = shared_memory + (tid % 16) * row_stride + (tid / 16) * 8;
    uint32_t smem = __cvta_generic_to_shared(shared_memory);

    // 四个 8 * 8 的矩阵加载一个 16 * 16 的矩阵
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(*reinterpret_cast<uint32_t*>(&reg.tiles[0][0].data[0])),
          "=r"(*reinterpret_cast<uint32_t*>(&reg.tiles[0][0].data[1])),
          "=r"(*reinterpret_cast<uint32_t*>(&reg.tiles[0][0].data[2])),
          "=r"(*reinterpret_cast<uint32_t*>(&reg.tiles[0][0].data[3]))
        : "r"(smem));
}

}  // namespace cuda_kernels::warp