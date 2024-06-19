import unittest
import torch
import context
from pybindings import flash_attn_fwd


def self_attention(q, k, v):
    score = q @ k.T / (q.shape[-1] ** 0.5)
    attention = torch.nn.functional.softmax(score)
    out = attention @ v

    return out


def flash_attn_v1(q, k, v, device='cuda', BLOCK_M=4):
    '''
    The tiny flash attention implement
    '''
    assert q.shape == k.shape
    assert q.shape == v.shape

    # Create output buffer in HBM.
    output_buffer = torch.zeros(v.shape, device=device)
    # Create denominator buffer in HBM.
    l = torch.zeros(v.shape[:-1], device=device)[..., None]
    # Create max(x) buffer in HBM.
    m = torch.ones(v.shape[:-1], device=device)[..., None] * -torch.inf

    Q_BLOCKS = torch.split(q, BLOCK_M, dim=-2)
    K_BLOCKS = torch.split(k, BLOCK_M, dim=-2)
    V_BLOCKS = torch.split(v, BLOCK_M, dim=-2)
    O_BLOCKS = list(torch.split(output_buffer, BLOCK_M, dim=-2))
    L_BLOCKS = list(torch.split(l, BLOCK_M, dim=-2))
    M_BLOCKS = list(torch.split(m, BLOCK_M, dim=-2))

    k_block_num = k.shape[-2] // BLOCK_M
    for j in range(k_block_num):
        kj = K_BLOCKS[j]
        vj = V_BLOCKS[j]

        q_block_num = q.shape[-2] // BLOCK_M
        for i in range(q_block_num):
            qi = Q_BLOCKS[i]
            old_o = O_BLOCKS[i]
            old_d = L_BLOCKS[i]
            old_m = M_BLOCKS[i]

            # Compute qk.
            x_qkt = (qi @ kj.T)
            # Get local max of qk.
            # keepdim to avoid auto squeeze.
            local_m = torch.max(x_qkt, dim=1, keepdim=True).values

            # # MatMul operator optimization version.
            # # Compute new max.
            # new_m = torch.maximum(old_m, local_m)
            # # Compute numerator. e^{x - max(x)}.
            # safe_e = torch.exp(x_qkt - new_m)
            # # Compute new part of denominator.
            # curr_d = torch.sum(safe_e, dim=1)[:, None]
            # # Update denominator.
            # new_d = old_d * torch.exp(old_m - new_m) + curr_d
            # # Update old output and accumulate new output.
            # new_o = old_o * torch.exp(old_m - new_m) * old_d / new_d + safe_e / new_d @ vj

            # Flash attention 1 with many redundant mul
            # Compute numerator. e^{x - max(x)}
            safe_e = torch.exp(x_qkt - local_m)
            # Compute new part of denominator.
            curr_d = torch.sum(safe_e, dim=1, keepdim=True)

            # Update max.
            new_m = torch.maximum(local_m, old_m)
            # Update denominator.
            new_d = old_d * torch.exp(old_m - new_m) + \
                curr_d * torch.exp(local_m - new_m)
            # Update old output and accumulate new output.
            new_o = (old_d * torch.exp(old_m - new_m) * old_o / new_d) + \
                (torch.exp(local_m - new_m) * safe_e / new_d @ vj.float())

            # Store new value.
            # NOTE:O_BLOCKS, L_BLOCKS, M_BLOCKS here will malloc addition memory
            L_BLOCKS[i] = new_d
            M_BLOCKS[i] = new_m
            O_BLOCKS[i] = new_o

    output_buffer = torch.cat(O_BLOCKS, dim=-2)

    return output_buffer


class TestFlashAttnFwdF32(unittest.TestCase):
    def test_flash_attn_fwd_v1(self):
        q = torch.randn(1, 1, 16, 64, device='cuda', dtype=torch.float32)
        k = torch.randn(1, 1, 16, 64, device='cuda', dtype=torch.float32)
        v = torch.randn(1, 1, 16, 64, device='cuda', dtype=torch.float32)
        o = torch.zeros(1, 1, 16, 64, device='cuda', dtype=torch.float32)
        print(o[0][0])
        flash_attn_fwd(q, k, v, o)
        ref_o = self_attention(q[0][0], k[0][0], v[0][0])
        print(o[0][0])
        print(ref_o)


if __name__ == '__main__':
    unittest.main()
