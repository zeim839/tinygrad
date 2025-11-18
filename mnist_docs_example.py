import numpy as np
from tinygrad.helpers import Timing
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from extra.datasets import fetch_mnist
import tinygrad
import os

# Enables CPU runtime.
os.environ["CPU"] = "1"

t1 = Tensor([1,2,3,4,5])
na = np.array([1,2,3,4,5])
t2 = Tensor(na)

print(t2)
print(t2.numpy())

# Tensor factory methods.
print("--- Tensor factor methods: ---")

# create a tensor of shape (2,3) filled with 5
full = Tensor.full(shape=(2,3), fill_value=5)
print(f"Full:\n{full.numpy()}\n\n")

# Create a tensor of shape (2,3) filled with 0.
zeros = Tensor.zeros(2,3)
print(f"Zeros:\n{zeros.numpy()}\n\n")

# Create a tensor of shape (2,3) filled with 1.
ones = Tensor.ones(2,3)
print(f"Ones:\n{ones.numpy()}\n\n")

# Create a tensor of the same shape as `full` filed with 2.
full_like = Tensor.full_like(full, fill_value = 2)
print(f"Full like:\n{full_like.numpy()}\n\n")

# Create a tensor with same shape as `full` with all zeros.
zeros_like = Tensor.zeros_like(full)
print(f"Zeros like:\n{zeros_like.numpy()}\n\n")

# Create a tensor with the same shape as `full` full of 1s.
ones_like = Tensor.ones_like(full)
print(f"Ones like:\n{ones_like.numpy()}\n\n")

# Create 3x3 identity matrix.
eye = Tensor.eye(3)
print(f"Eye (identity matrix):\n{eye.numpy()}\n\n")

# Create a tensor of shape (10,) filled with values from 0 to 9.
arange = Tensor.arange(start=0, stop=10, step=1)
print(f"Arange:\n{arange.numpy()}\n\n")

# --- Random values. --

# create a tensor of shape (2, 3) filled with random values from a uniform distribution
rand = Tensor.rand(2,3)
print(f"Rand (uniform):\n{rand.numpy()}\n\n")

randn = Tensor.randn(2, 3)
print(f"Rand (normal):\n{randn.numpy()}\n\n")

# -- Tensors with custom datatypes ---
t3 = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)
print(f"t3 tensor dtype: {t3.dtype}")

# -- Tensor binary ops --
t4 = Tensor([1, 2, 3, 4, 5])
t5 = (t4 + 1) * 2
t6 = (t5 * t4).relu().log_softmax()

print(f"Result of a bunch of tensor ops: {t6.numpy()}")

# -- Models --

# MNIST digit classificaiton.

class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leaky_relu()
        x = self.l2(x)
        return x

net = TinyNet()
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

X_train, Y_train, X_test, Y_test = fetch_mnist()

with Tensor.train():
    for step in range(1000):

        # Random sample a batch.
        samp = np.random.randint(0, X_train.shape[0], size=(64))
        batch = Tensor(X_train[samp], requires_grad = False)

        # Get the corresponding batch labels.
        labels = Tensor(Y_train[samp])

        # Forward pass.
        out = net(batch)

        # Compute loss.
        loss = out.sparse_categorical_crossentropy(labels)

        # Zero gradients.
        opt.zero_grad()

        # Backward pass.
        loss.backward()

        # Update parameters.
        opt.step()

        # Calculate accuracy
        pred = out.argmax(axis=-1)
        acc = (pred == labels).mean()

        if step % 100 == 0:
            print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

@TinyJit
def jit(x):
    return net(x).realize()

with Timing("Time: "):
    avg_acc = 0
    for step in range(1000):

        # Random sample a batch.
        samp = np.random.randint(0, X_test.shape[0], size=(64))
        batch = Tensor(X_test[samp], requires_grad=False)

        # Get the corresponding batch labels.
        labels = Y_test[samp]

        # Forward pass.
        out = jit(batch)

        # Calculate accuracy.
        pred = out.argmax(axis=-1).numpy()
        avg_acc += (pred == labels).mean()

    print(f"Test Accuracy: {avg_acc / 1000}")
