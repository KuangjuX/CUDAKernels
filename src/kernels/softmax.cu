#include "kernels/softmax.hpp"
#include "warp/mod.hpp"

namespace cuda_kernels::kernels {

template <typename Element, const int THREAD_NUMS, const int WARP_SIZE>
__global__ void softmax_kernel(const Element* x, Element* y, Element* exp_total,
                               const int size) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    constexpr int WARP_NUMS = (THREAD_NUMS + WARP_SIZE - 1) / WARP_SIZE;

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    extern __shared__ Element shared[WARP_NUMS];

    Element exp_sum = (idx < size) ? expf(x[idx]) : 0;

    // 使用 Warp Reduce 计算每个 warp 的指数和
    exp_sum = warp::warp_reduce_sum<Element, WARP_SIZE>(exp_sum);

    // `lane_id` = 0 表示最终的 reduce sum 结果
    if (lane_id == 0) {
        shared[warp_id] = exp_sum;
        // printf("block: %d, warp: %d, sum: %f\n", blockIdx.x, warp_id,
        // exp_sum);
    }

    // 同步一个 thread block 里的线程
    __syncthreads();

    // 使用一个 warp 计算一个 thread block 的和
    exp_sum = (lane_id < WARP_NUMS) ? shared[lane_id] : 0;

    if (warp_id == 0) {
        exp_sum = warp::warp_reduce_sum<Element, WARP_NUMS>(exp_sum);
        // printf("block: %d, sum: %f\n", blockIdx.x, exp_sum);
    }

    // 计算所有 thread blocks 的指数和
    if (tid == 0) atomicAdd(exp_total, exp_sum);

    // 同步所有 grid，确保所有 thread blocks 计算完
    __threadfence();

    // 计算每个线程的 softmax 结果
    if (idx < size) {
        y[idx] = expf(x[idx]) / (*exp_total);
    }
}

void softmax(const torch::Tensor& input, torch::Tensor& output, int64_t size) {
    const int THREAD_SIZE = 1024;
    const int WARP_SIZE = 32;
    int block_size = (size + THREAD_SIZE - 1) / THREAD_SIZE;

    if (input.dtype() == torch::kFloat32) {
        float* exp_total;
        cudaMalloc(&exp_total, sizeof(float));
        cudaMemset(exp_total, 0, sizeof(float));
        softmax_kernel<float, THREAD_SIZE, WARP_SIZE>
            <<<block_size, THREAD_SIZE>>>(input.data_ptr<float>(),
                                          output.data_ptr<float>(), exp_total,
                                          size);
        cudaFree(exp_total);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}

}  // namespace cuda_kernels::kernels