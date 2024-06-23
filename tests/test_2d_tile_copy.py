import unittest
import torch
import context
from pybindings import copy_2d_tensor_g2r


class Test2DTileCopy(unittest.TestCase):
    def test_vec_copy_f32_v1(self):
        # Create a 16 * 16 tensor
        input = torch.tensor([float(i) for i in range(256)], device='cuda')
        output = torch.zeros(256, device='cuda')
        height = 16
        width = 16
        copy_2d_tensor_g2r(input, output, height, width)


if __name__ == '__main__':
    unittest.main()
