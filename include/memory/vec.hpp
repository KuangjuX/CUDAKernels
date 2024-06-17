#pragma once

#include "common.hpp"
#include "cuda_utils.hpp"
#include "memory/types/register.hpp"

namespace cuda_kernels::memory {

template <typename Element>
__device__ inline static void vec_load_g2s(Element* src, Element* dst,
                                           int length) {
    constexpr int element_per_transfer = sizeof(float4) / sizeof(Element);
    constexpr int total_calls = length / element_per_transfer;
    int tid = threadIdx.x;

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    __syncwarp();

#pragma unroll
    for (int i = lane_id; i < total_calls; i += WARP_SIZE) {
        if (i * element_per_transfer < length) {
            *(float4*)dst[i * element_per_transfer] =
                *(float4*)src[i * element_per_transfer];
        }
    }
}

template <typename Element>
__device__ inline static void vec_store_s2g(Element* src, Element* dst,
                                            int length) {
    constexpr int element_per_transfer = sizeof(float4) / sizeof(Element);
    constexpr int total_calls = length / element_per_transfer;

    int tid = threadIdx.x;

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    __syncwarp();

#pragma unroll
    for (int i = lane_id; i < total_calls; i += WARP_SIZE) {
        if (i * element_per_transfer < length) {
            *(float4*)dst[i * element_per_transfer] =
                *(float4*)src[i * element_per_transfer];
        }
    }
}

// Reference:
// https://github.com/HazyResearch/ThunderKittens/blob/main/src/ops/warp/memory/vec/global_to_register.cuh
// TODO: Print to debug the code
/**
 * @brief Load data into a register vector from a source array in global memory.
 * @param[out] dst The register vector to load into.
 * @param[in] src The source array in global memory.
 */
template <const int OUTER_LENGTH>
__device__ inline static void vec_load_g2r_f32(
    const float* src, types::Register<float2, OUTER_LENGTH>& dst) {
    // 使用 warp 进行协作式加载
    int lane_id = threadIdx.x % WARP_SIZE;
    __syncwarp();

#pragma unroll
    for (auto w = 0; w < (dst.outer_dim + 1) / 2; ++w) {
        int idx = w * 32 + (lane_id % 4) * 8 + lane_id / 4;
        int o_dim = w * 2 + (lane_id % 4) / 2;
        // This should be a maximally coalesced load.
        if (idx < dst.outer_dim * 16) {
            if (lane_id % 2 == 0)
                dst[o_dim].x = src[idx];
            else
                dst[o_dim].y = src[idx];
        }
    }

    __syncwarp();

    // Now we need to do a bunch of shuffle_sync's to make sure everyone has
    // everything they need.

    // 相邻的四个线程进行数据的同步，最后一个 warp
    // 中相邻四个线程里的数据是一样的。
#pragma unroll
    for (auto w = 0; w < dst.outer_dim; w++) {
        int leader = (lane_id / 4) * 4 + 2 * (w % 2);
        dst[w].x = __shfl_sync(0xffffffff, dst[w].x, leader);
        dst[w].y = __shfl_sync(0xffffffff, dst[w].y, leader + 1);
    }
}

// Reference:
// https://github.com/HazyResearch/ThunderKittens/blob/main/src/ops/warp/memory/vec/global_to_register.cuh
// TODO: Print to debug the code
/**
 * @brief Store data from a register vector to a destination array in global
 * memory.
 * @param[in] src The register vector to store from.
 * @param[out] dst The destination array in global memory.
 */
template <const int OUTER_LENGTH>
__device__ inline static void vec_store_r2g_f32(
    types::Register<float2, OUTER_LENGTH>& src, float* dst) {
    // 使用 warp 进行协作式存储
    int lane_id = threadIdx.x % WARP_SIZE;
    __syncwarp();

#pragma unroll
    for (auto w = 0; w < (src.outer_dim + 1) / 2; ++w) {
        int idx = w * 32 + (lane_id % 4) * 8 + lane_id / 4;
        int o_dim = w * 2 + (lane_id % 4) / 2;
        // This should be a maximally coalesced load.
        if (idx < src.outer_dim * 16) {
            if (lane_id % 2 == 0)
                dst[idx] = src[o_dim].x;
            else
                dst[idx] = src[o_dim].y;
        }
    }
}

}  // namespace cuda_kernels::memory