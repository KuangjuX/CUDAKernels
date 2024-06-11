import unittest
import torch
import context
from pybindings import reduce_sum, reduce_max


class TestReduceSum(unittest.TestCase):
    def test_reduce_sum_v1(self):
        input = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device='cuda')
        output = torch.zeros(1, device='cuda')
        size = input.size(0)
        reduce_sum(input, output, size)
        self.assertEqual(output.item(), input.sum().item())

    def test_reduce_sum_v2(self):
        input = torch.tensor([float(i) for i in range(1024)], device='cuda')
        output = torch.zeros(1, device='cuda')
        size = input.size(0)
        reduce_sum(input, output, size)
        self.assertEqual(output.item(), input.sum().item())

    def test_reduce_sum_v3(self):
        input = torch.tensor([float(i) for i in range(4096)], device='cuda')
        output = torch.zeros(1, device='cuda')
        size = input.size(0)
        reduce_sum(input, output, size)
        self.assertEqual(output.item(), input.sum().item())

    def test_reduce_max_v1(self):
        input = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7], device='cuda', dtype=torch.int32)
        output = torch.zeros(1, device='cuda', dtype=torch.int32)
        size = input.size(0)
        reduce_max(input, output, size)
        self.assertEqual(output.item(), input.max().item())

    def test_reduce_max_v2(self):
        input = torch.randint(
            0, 100, (1024,), device='cuda', dtype=torch.int32)
        output = torch.zeros(1, device='cuda', dtype=torch.int32)
        size = input.size(0)
        reduce_max(input, output, size)
        self.assertEqual(output.item(), input.max().item())

    def test_reduce_max_v3(self):
        input = torch.randint(
            0, 100, (4096,), device='cuda', dtype=torch.int32)
        output = torch.zeros(1, device='cuda', dtype=torch.int32)
        size = input.size(0)
        reduce_max(input, output, size)
        self.assertEqual(output.item(), input.max().item())


if __name__ == '__main__':
    unittest.main()
