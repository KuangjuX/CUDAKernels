import unittest
import torch
import context
from pybindings import reduce_sum


class TestReduceSum(unittest.TestCase):
    def test_reduce_sum(self):
        input = torch.randn(10, device='cuda')
        output = torch.zeros(1, device='cuda')
        size = input.size(0)
        reduce_sum(input, output, size)
        self.assertEqual(output.item(), input.sum().item())


if __name__ == '__main__':
    unittest.main()
