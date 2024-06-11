import unittest
import torch
import context
from pybindings import softmax


class TestSoftmax(unittest.TestCase):
    def test_softmax_v1(self):
        input = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device='cuda')

        output = torch.zeros(9, device='cuda')
        size = input.size(0)
        softmax(input, output, size)
        self.assertTrue(torch.allclose(
            output, torch.nn.functional.softmax(input, dim=0), atol=1e-2))

    def test_softmax_v2(self):
        input = torch.randn(1024, device='cuda')
        output = torch.zeros(1024, device='cuda')
        size = input.size(0)
        softmax(input, output, size)
        self.assertTrue(torch.allclose(
            output, torch.nn.functional.softmax(input, dim=0), atol=1e-2))

    def test_softmax_v3(self):
        input = torch.randn(4096, device='cuda')
        output = torch.zeros(4096, device='cuda')
        size = input.size(0)
        softmax(input, output, size)
        self.assertTrue(torch.allclose(
            output, torch.nn.functional.softmax(input, dim=0), atol=1e-2))


if __name__ == '__main__':
    unittest.main()
