#include "kernels/mod.hpp"
#include "warp/mod.hpp"

#include <type_traits>

namespace cuda_kernels::kernels {

template <typename Element, const int THREAD_NUMS, const int WARP_SIZE>
__global__ void reduce_sum_kernel(const Element* input, Element* output,
                                  int size) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // always <= 32 warps per block(limited by 1024 threads per block)
    constexpr int WARP_NUM = (THREAD_NUMS + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ Element shared[WARP_NUM];

    // 得到当前线程应该处理的数据
    Element sum = (idx < size) ? input[idx] : 0;

    // 使用 warp 做累加求和
    sum = warp::warp_reduce_sum<Element, WARP_SIZE>(sum);

    // 取 lane_id 为 0 的结果为该 warp 处理的结果
    if (lane_id == 0) {
        shared[warp_id] = sum;
        // printf("block: %d, warp: %d, sum: %f\n", blockIdx.x, warp_id, sum);
    }

    __syncthreads();

    // WARP_NUM 应该小于 32，因此可以使用一个 warp 来计算所有 warps 的结果
    sum = (lane_id < WARP_NUM) ? shared[lane_id] : 0;

    // 使用一个 warp 对所有 warps 计算出的结果做累加
    if (warp_id == 0) {
        sum = warp::warp_reduce_sum<Element, WARP_NUM>(sum);
    }

    // 对所有 blocks 进行累加得到 reduce 的结果
    if (tid == 0) atomicAdd(output, sum);
}

template <typename Element, const int THREAD_NUMS, const int WARP_SIZE>
__global__ void reduce_max_kernel(const Element* input, Element* output,
                                  int size) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    constexpr int WARP_NUM = (THREAD_NUMS + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ Element shared[WARP_NUM];

    // 得到当前线程应该处理的数据
    Element max = (idx < size) ? input[idx] : 0;

    // 使用 warp 做 reduce 求最大值
    max = warp::warp_reduce_max<Element, WARP_SIZE>(max);

    // 取 lane_id 为 0 的结果为该 warp 处理的结果
    if (lane_id == 0) {
        shared[warp_id] = max;
    }

    __syncthreads();

    // WARP_NUM 应该小于 32，因此可以使用一个 warp 来计算所有 warps 的结果
    max = (lane_id < WARP_NUM) ? shared[lane_id] : 0;

    // 使用一个 warp 对所有 warps 计算出的结果做 reduce max
    if (warp_id == 0) {
        max = warp::warp_reduce_max<Element, WARP_NUM>(max);
    }

    // 对所有 blocks 进行 reduce max 得到结果
    if (tid == 0) {
        if (std::is_integral_v<Element>) {
            atomicMax(output, max);
        } else if (std::is_floating_point_v<Element>) {
            // TODO: atomicMax is not supported for floating point types
        }
    }
}

void reduce_sum(const torch::Tensor& input, torch::Tensor& output,
                int64_t size) {
    const int THREAD_SIZE = 1024;
    const int WARP_SIZE = 32;
    int block_size = (size + THREAD_SIZE - 1) / THREAD_SIZE;

    if (input.dtype() == torch::kFloat32) {
        reduce_sum_kernel<float, THREAD_SIZE, WARP_SIZE>
            <<<block_size, THREAD_SIZE>>>(input.data_ptr<float>(),
                                          output.data_ptr<float>(), size);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

void reduce_max(const torch::Tensor& input, torch::Tensor& output,
                int64_t size) {
    const int THREAD_SIZE = 1024;
    const int WARP_SIZE = 32;
    int block_size = (size + THREAD_SIZE - 1) / THREAD_SIZE;

    if (input.dtype() == torch::kInt32) {
        reduce_max_kernel<int, THREAD_SIZE, WARP_SIZE>
            <<<block_size, THREAD_SIZE>>>(input.data_ptr<int>(),
                                          output.data_ptr<int>(), size);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

}  // namespace cuda_kernels::kernels