import unittest
import torch
import context
from pybindings import flash_attn_fwd


class TestFlashAttnFwdF32(unittest.TestCase):
    def test_flash_attn_fwd_v1(self):
        q = torch.randn(16, 16, 16, 64, device='cuda', dtype=torch.float32)
        k = torch.randn(16, 16, 16, 64, device='cuda', dtype=torch.float32)
        v = torch.randn(16, 16, 16, 64, device='cuda', dtype=torch.float32)
        o = torch.zeros(16, 16, 16, 64, device='cuda', dtype=torch.float32)
        flash_attn_fwd(q, k, v, o)
        # print(o)
        # Print the output tensor[0][0]
        print(o[0][0])


if __name__ == '__main__':
    unittest.main()
