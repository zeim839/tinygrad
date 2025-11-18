import numpy as np
from tinygrad import Tensor

x = Tensor.empty(4,4, dtype='int')
y = Tensor.empty(4,4, dtype='int')
z = x + y

print(z.schedule())
