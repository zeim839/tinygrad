import os
import timeit
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.datasets import mnist

# Use the CPU.
os.environ["CPU"] = "1"
os.environ["METAL"] = "0"

# This is the model used in the Keras tutorial.
class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))
        x = self.l2(x).relu().max_pool2d((2,2))
        return self.l3(x.flatten(1).dropout(0.5))

X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)

# Instantiate the model.
model = Model()

# Initialize Adam optimizer.
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128

# Training step.
def step():

    # Allows dropout regularization to work.
    Tensor.training = True

    # Sample random batch.
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]

    # Reset stored gradients.
    optim.zero_grad()

    # Calculate loss and accumulate new gradients.
    loss = model(X).sparse_categorical_crossentropy(Y).backward()

    # Apply gradients.
    optim.step()

    return loss

# Time a step (benchmark).
print("Timing (without JIT):\n", timeit.repeat(step, repeat=5, number=1))

# Time a step with JIT.
jit_step = TinyJit(step)
print("Timing (with JIT):\n", timeit.repeat(jit_step, repeat=5, number=1))

# Train the network.
for step in range(7000):
    loss = jit_step()
    if step % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
