import torch
import torch.utils.benchmark as benchmark
import context
from pybindings import flash_attn_fwd
import os


def self_attention(q, k, v):
    score = q @ k.T / (q.shape[-1] ** 0.5)
    attention = torch.nn.functional.softmax(score)
    out = attention @ v

    return out


def bench_flash_attn_f32(n, d):
    q = torch.randn(1, 1, n, d, dtype=torch.float32, device='cuda')
    k = torch.randn(1, 1, n, d, dtype=torch.float32, device='cuda')
    v = torch.randn(1, 1, n, d, dtype=torch.float32, device='cuda')
    o = torch.zeros(1, 1, n, d, dtype=torch.float32, device='cuda')

    t0 = benchmark.Timer(
        stmt='self_attention(q, k, v)',
        setup='from __main__ import self_attention',
        globals={'q': q[0][0], 'k': k[0][0], 'v': v[0][0]},
    )

    t1 = benchmark.Timer(
        stmt='flash_attn_fwd(q, k, v, o)',
        setup='from pybindings import flash_attn_fwd',
        globals={'q': q, 'k': k, 'v': v, 'o': o},
    )

    print('n =', n, 'd =', d)
    print(t0.timeit(100))
    print(t1.timeit(100))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    # bench_flash_attn_f32(256, 256)
    # bench_flash_attn_f32(512, 512)
    # bench_flash_attn_f32(1024, 1024)
    # bench_flash_attn_f32(2048, 2048)

    # bench_flash_attn_f32(16, 1024)
    # bench_flash_attn_f32(64, 1024)
    bench_flash_attn_f32(64, 4096)
