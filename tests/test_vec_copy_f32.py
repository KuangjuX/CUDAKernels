import unittest
import torch
import context
from pybindings import vec_copy_tensor_g2r


class TestVecCopyF32(unittest.TestCase):
    def test_vec_copy_f32_v1(self):
        # input = torch.randn(1024, device='cuda', dtype=torch.float32)
        input = torch.tensor([float(i) for i in range(4096)], device='cuda')
        output = torch.zeros(4096, device='cuda', dtype=torch.float32)
        vec_copy_tensor_g2r(input, output, 4096)
        # for i in range(2048):
        #     print(f'{input[i]} -> {output[i]}')
        self.assertTrue(torch.allclose(input, output))


if __name__ == '__main__':
    unittest.main()
