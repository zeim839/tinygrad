from tinygrad import Tensor
from tinygrad.device import Device
assert Device.DEFAULT == "METAL"

x = Tensor.ones(1)
y = Tensor.ones(1)
z = x + y

z.tolist()
