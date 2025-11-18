from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn import datasets

# Define first model.
class LinearModel:
    def __init__(self):
        self.l1 = nn.Linear(28*28, 512, bias=True)
        self.l2 = nn.Linear(512, 10, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = x / 255
        x = self.l1(x).relu()
        x = self.l2(x).softmax()
        return x

class LeakyModel:
    def __init__(self):
        self.l1 = nn.Linear(28*28, 512, bias=False)
        self.l2 = nn.Linear(512, 10, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).leaky_relu()
        x = self.l2(x)
        return x

# Download Dataset.
X_train, Y_train, X_test, Y_test = datasets.mnist()
X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

# Inspect dataset shapes.
print("Dataset Shape:")
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype, Y_test.shape, Y_test.dtype, "\n")

# Instantiate models.
linearModel = LinearModel()
leakyModel = LeakyModel()

# Instantiate optimizer.
linearAdam = Adam(nn.state.get_parameters(linearModel))
leakyAdam = Adam(nn.state.get_parameters(leakyModel))

# Hyperparameters
batch_size = 128

# Training step.
@TinyJit
def step():
    Tensor.training = True

    # Sample random batch.
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]

    # Reset stored gradients.
    linearAdam.zero_grad()
    leakyAdam.zero_grad()

    # Calculate loss and accumulate new gradients.
    linearLoss = linearModel(X).sparse_categorical_crossentropy(Y).backward()
    leakyLoss = leakyModel(X).sparse_categorical_crossentropy(Y).backward()

    # Apply gradients.
    linearAdam.step()
    leakyAdam.step()

    return linearLoss, leakyLoss

# Train the network.
print("Model Training:")
for step_ in range(1000):
    linearLoss, leakyLoss = step()
    if step_ % 100 == 0:
        Tensor.training = False
        linacc = (linearModel(X_test).argmax(axis=1) == Y_test).mean().item()
        leakyacc = (leakyModel(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"Step {step_:4d} | LinLoss {linearLoss.item():.2f} | LeakyLoss {leakyLoss.item():.2f} | LinAcc {linacc:.2f} | LeakyAcc {leakyacc:.2f}")

# Evaluation.
print("\nEvaluation:")
linearMean, leakyMean = 0, 0
for i in range(1000):
    samples = Tensor.randint(batch_size, high=X_test.shape[0])
    X, Y = X_test[samples], Y_test[samples]

    linearAcc = (linearModel(X).argmax(axis=1) == Y).mean().item()
    leakyAcc = (leakyModel(X).argmax(axis=1) == Y).mean().item()

    linearMean += linearAcc
    leakyMean += leakyAcc

print(f"Linear Mean Accuracy: {linearMean / 1000}")
print(f"Leaky Mean Accuracy: {leakyMean / 1000}")
