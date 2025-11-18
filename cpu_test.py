from tinygrad import Tensor
from tinygrad.helpers import Timing

x = Tensor.ones(9000,9000, device="cpu")
y = Tensor.rand(9000,9000, device="cpu")
z = x @ y

with Timing(prefix="Total ", on_exit=lambda x: f"{x}"):
    z.realize()
