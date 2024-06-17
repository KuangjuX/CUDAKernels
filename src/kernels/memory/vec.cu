#include "kernels/memory/vec.hpp"
#include "memory/mod.hpp"

namespace cuda_kernels::kernels {

template <int OUTER_LENGTH>
__global__ void vec_copy_g2r_kernel_f32(const float* src, float* dst,
                                        int length) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int offset = warp_id * OUTER_LENGTH * 32 / 2;
    memory::types::Register<float2, OUTER_LENGTH> regs;

    memory::vec_load_g2r_f32<OUTER_LENGTH>(src + offset, regs);

    // if (tid == 0) {
    //     regs.print(tid);
    // } else if (tid == 4) {
    //     regs.print(tid);
    // } else if (tid == 32) {
    //     regs.print(tid);
    // } else if (tid == 64) {
    //     regs.print(tid);
    // } else if (tid == 96) {
    //     regs.print(tid);
    // }

    // __syncthreads();

    if (tid == 0) {
        regs.print(tid);
    }
    __syncthreads();
    if (tid == 32) {
        regs.print(tid);
    }
    __syncthreads();
    if (tid == 64) {
        regs.print(tid);
    }
    __syncthreads();
    if (tid == 96) {
        regs.print(tid);
    }

    memory::vec_store_r2g_f32<OUTER_LENGTH>(regs, dst + offset);
}

void vec_copy_g2r(const torch::Tensor& input, torch::Tensor& output,
                  int64_t size) {
    const int OUTER_LENGTH = 32;
    const int THREAD_SIZE = size * 2 / OUTER_LENGTH;

    auto dtype = input.dtype();

    if (dtype == torch::kFloat32) {
        vec_copy_g2r_kernel_f32<OUTER_LENGTH><<<1, THREAD_SIZE>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), size);
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}
}  // namespace cuda_kernels::kernels