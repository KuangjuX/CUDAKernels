import torch
import torch.utils.benchmark as benchmark
import context
from pybindings import flash_attn_fwd


def self_attention(q, k, v):
    score = q @ k.T / (q.shape[-1] ** 0.5)
    attention = torch.nn.functional.softmax(score)
    out = attention @ v

    return out


def bench_flash_attn_f32():
    q = torch.randn(1, 1, 1024, 1024, dtype=torch.float32, device='cuda')
    k = torch.randn(1, 1, 1024, 1024, dtype=torch.float32, device='cuda')
    v = torch.randn(1, 1, 1024, 1024, dtype=torch.float32, device='cuda')
    o = torch.zeros(1, 1, 1024, 1024, dtype=torch.float32, device='cuda')

    # ref_o = self_attention(q[0][0], k[0][0], v[0][0])
    # print(ref_o)
    # flash_attn_fwd(q, k, v, o)
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

    print('Print the time taken for 100 runs of the function in seconds:')
    print(t0.timeit(100))
    print(t1.timeit(100))


if __name__ == '__main__':
    bench_flash_attn_f32()
