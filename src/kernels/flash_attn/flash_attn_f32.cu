#include "kernels/flash_attn/flash_attn_f32.hpp"

#include <stdio.h>

namespace cuda_kernels::kernels {
/**
 * @brief Flash Attention forward kernel
 * @param[in] Q Query tensor
 * @param[in] K Key tensor
 * @param[in] V Value tensor
 * @param[in] N Batch size
 * @param[in] d Multi-head dimension
 * @param[in] Tc Sequence length
 * @param[in] Tr Sequence length
 * @param[in] Bc Block size
 * @param[in] Br Block size
 * @param[in] softmax_scale Softmax scale
 * @param[out] l softmax sum tensor
 * @param[out] m softmax max tensor
 * @param[out] O Output tensor
 */
__global__ void flash_attn_fwd_f32_kernel(
    const float* Q, const float* K, const float* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    const float softmax_scale, float* l, float* m, float* O) {
    int tid = threadIdx.x;
    // Batch, head 在 thread blocks 上进行并行
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 一个 thread block 处理一个 softmax(QK^T)V

    // 获得 Q, K, V 的偏移
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    // 获得 l, m 的偏移
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    // 为 Q，K，V，S 定义 SRAM
    extern __shared__ float sram[];

    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = Qi + tile_size;
    float* Vj = Kj + tile_size;
    float* S = Vj + tile_size;

    // 外层循环将 K, V 进行 tile 并加载到 SRAM 中
    for (int j = 0; j < Tc; ++j) {
        // 加载 Kj, Vj 到 SRAM
        for (int x = 0; x < d; ++x) {
            // 看起来没有进行内存合并访问？
            Kj[(tid * d) + x] = K[qkv_offset + (tile_size * j) + (tid * d) + x];
            Vj[(tid * d) + x] = V[qkv_offset + (tile_size * j) + (tid * d) + x];
        }

        // for (int x = 0; x < d; ++x) {
        //     // 打印 Kj, Vj
        //     printf("Kj[%d]: %f, Vj[%d]: %f\n", (tid * d) + x, Kj[(tid * d) +
        //     x],
        //            (tid * d) + x, Vj[(tid * d) + x]);
        // }

        // 同步所有线程，内层循环可以正确使用 Kj, Vj
        __syncthreads();

        // 内层循环
        for (int i = 0; i < Tr; ++i) {
            // 加载 Qi 到 SRAM，l，m 到寄存器
            for (int x = 0; x < d; ++x) {
                // 每个线程处理一行(一个 d)
                Qi[(tid * d) + i] =
                    Q[qkv_offset + (tile_size * i) + (tid * d) + i];
            }

            // for (int x = 0; x < d; ++x) {
            //     // 打印 Qi
            //     printf("Qi[%d]: %f\n", (tid * d) + i, Qi[(tid * d) + i]);
            // }
            // 一次循环加载一次 l，m
            float row_m_prev = m[lm_offset + (Br * i) + tid];
            float row_l_prev = l[lm_offset + (Br * i) + tid];

            // S = Qk^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            // 循环一个切块的大小
            for (int y = 0; y < Bc; ++y) {
                float sum = 0;
                // S = Qk^T
                // 一个线程处理一个 d，每次都和 K 进行计算
                for (int x = 0; x < d; ++x) {
                    sum += Qi[(tid * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tid) + y] = sum;

                if (sum > row_m) row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; ++y) {
                S[(Bc * tid) + y] = __expf(S[(Bc * tid) + y] - row_m);
                row_l += S[(Bc * tid) + y];
            }

            // 打印 S
            // for (int y = 0; y < Bc; ++y) {
            //     printf("S[%d]: %f\n", (Bc * tid) + y, S[(Bc * tid) + y]);
            // }

            // 计算新的 m 和 l
            float row_m_new = max(row_m, row_m_prev);
            // online softmax 的计算
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

            // printf("row_m_new: %f, row_l_new: %f\n", row_m_new, row_l_new);

            // 将 O,l,m 写回到 HBM
            for (int x = 0; x < d; ++x) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; ++y) {
                    pv += S[(Bc * tid) + y] * Vj[(y * d) + x];
                }
                // 同样是基于迭代式对 O 进行更新
                O[qkv_offset + (tile_size * i) + (tid * d) + x] =
                    (1 / row_l_new) *
                        ((row_l_prev * __expf(row_m_prev - row_m_new))) *
                        O[qkv_offset + (tile_size * i) + (tid * d) + x] +
                    (__expf(row_m - row_m_new) * pv);
            }

            // 打印 O
            // for (int x = 0; x < d; ++x) {
            //      printf("O[%d]: %f\n", (tid * d) + x,
            //           O[qkv_offset + (tile_size * i) + (tid * d) + x]);
            // }

            m[lm_offset + (Br * i) + tid] = row_m_new;
            l[lm_offset + (Br * i) + tid] = row_l_new;
        }
        // 同步内层循环
        __syncthreads();
    }
}

void flash_attn_fwd(const torch::Tensor& Q, const torch::Tensor& K,
                    const torch::Tensor& V, torch::Tensor& O) {
    // 设置块大小，其中 Bc = ceil(M/4d), Br = min(ceil(M/4d), d)
    // 这里全部使用 32 作为块大小
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    printf("B: %d, nh: %d, N: %d, d: %d\n", B, nh, N, d);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto type = Q.dtype();

    // auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, torch::kFloat32);
    auto m = torch::full({B, nh, N}, -FP_INFINITE, torch::kFloat32);
    torch::Device device(torch::kCUDA);

    l = l.to(device);
    m = m.to(device);

    // 计算 SRAM 的大小
    const int sram_size =
        (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock,
                           0);
    printf("Max shared memory per block: %d, requested shared memory: %d\n",
           max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    if (type == torch::kFloat32) {
        flash_attn_fwd_f32_kernel<<<grid_dim, block_dim, sram_size>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d,
            Tc, Tr, Bc, Br, softmax_scale, l.data_ptr<float>(),
            m.data_ptr<float>(), O.data_ptr<float>());
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}
}  // namespace cuda_kernels::kernels