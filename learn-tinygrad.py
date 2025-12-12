from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import mv_address
from tinygrad.uop import Ops
import unittest, itertools

# Explanation of UOps:
# https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250119_uop_singleton.md

# -----  tensor.py - staticmethods ------
class TensorStaticMethods(unittest.TestCase):

    # Empty is a len x dtype_size buffer with some shape.
    def test_empty(self):
        UOp.unique_num = itertools.count(0)
        empty = Tensor.empty(3,3)
        self.assertEqual(empty.shape, (3,3))

        buff = UOp(Ops.BUFFER, dtypes.float, arg=9, src=(
            UOp(Ops.UNIQUE, dtypes.void, arg=0, src=()),
            UOp(Ops.DEVICE, dtypes.void, arg='CPU', src=()),
        )).reshape(3,3)

        self.assertEqual(empty.uop, buff)

    # Creates an empty tensor, then take's the tensor's buffer and
    # allocate's it such that it points to the blob. The blob must
    # live for as long as the new tensor.
    def test_from_blob(self):
        UOp.unique_num = itertools.count(0)
        x = memoryview(bytearray(16)).cast('I')
        tensor = Tensor.from_blob(mv_address(x), (1, 5))
        self.assertEqual(tensor.shape, (1,5))

    # Creates a tensor from an internet URL.
    # Downloads into some temp directory, then creates a Tensor whose
    # buffer points to the file on the disk.
    def test_from_url(self):
        UOp.unique_num = itertools.count(0)
        Tensor.from_url("https://ufosc.org")

    # Rand is very complicated. I look at it later.
    def test_rand(self):
        UOp.unique_num = itertools.count(0)
        x = Tensor.rand(2,3)

    # Creates a tensor of the given shape filled with a constant value
    # Creates a single-value constant of the specified value (e.g. 50.0).
    # Reshapes constant into num_dim*(1,).
    # Expands the dimensions to match the specified shape.
    def test_full(self):
        UOp.unique_num = itertools.count(0)
        x = Tensor.full((2,3), 50, dtype=dtypes.float)
        self.assertEqual(x.shape, (2,3))

        y = UOp(Ops.CONST, dtypes.float, arg=50.0, src=(
            UOp(Ops.DEVICE, dtypes.void, arg='CPU'),
            UOp(Ops.UNIQUE, dtypes.void, arg=0)
        )).reshape(1,1).expand(2,3)

        self.assertEqual(x.uop, y)

    # Tensor.full with constant value 0.
    def test_zeros(self):
        UOp.unique_num = itertools.count(0)
        y = UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
            UOp(Ops.DEVICE, dtypes.void, arg='CPU'),
            UOp(Ops.UNIQUE, dtypes.void, arg=0)
        )).reshape(1,1,1).expand(3,3,3)

        x = Tensor.zeros(3,3,3, dtype=dtypes.float)
        self.assertEqual(x.uop, y)

    # Tensor.full with constant value 1.
    def test_ones(self):
        UOp.unique_num = itertools.count(0)
        y = UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
            UOp(Ops.DEVICE, dtypes.void, arg='CPU'),
            UOp(Ops.UNIQUE, dtypes.void, arg=0)
        )).reshape(1,1,1).expand(3,3,3)

        x = Tensor.ones(3,3,3, dtype=dtypes.float)
        self.assertEqual(x.uop, y)

if __name__ == "__main__":
    unittest.main()
