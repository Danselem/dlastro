import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import os

os.makedirs("./plots", exist_ok=True)

# Create a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)
print(type(x))

# Create a NumPy array
y = np.array([4.0, 5.0, 6.0])
print(y)
print(type(y))

# Convert NumPy array to PyTorch tensor
x = torch.from_numpy(y)
print(x)
print(type(x))

# Convert PyTorch tensor back to NumPy array
y = x.numpy()
print(y)
print(type(y))

x = torch.tensor(2.0)
print(x)
print(x.shape)


x = torch.tensor(2.0, requires_grad=True)
print(x)

# Define our function y = sin(x) * x^2 using torch operations
def f(x): 
    return torch.sin(x) * x**2

x = torch.tensor(1.234, requires_grad=True)
y = f(x)
print(f"y = {y.item()}")

# Do the backward pass through f(x)
y.backward()
print("dy/dx =", x.grad.item())

# Must detach the PyTorch tensor before converting to a NumPy array if it is tracking gradients
x_val = x.detach().numpy()
analytic = 2 * x_val * np.sin(x_val) + (x_val**2) * np.cos(x_val)

print(f"autodiff : {x.grad}")
print(f"analytic : {analytic}")
print(f"difference: {abs(x.grad.numpy() - analytic)}")


# Redefine x and its gradient
x = torch.tensor(1.234, requires_grad=True)

# First backward pass (computes dy/dx)
f(x).backward()
print("after first backward:", x.grad)

# Second backward pass (computes dy/dx again and adds to existing gradient)
f(x).backward()
print("after second backward:", x.grad)

# Clear existing gradient and compute again
x.grad.zero_()
f(x).backward()
print("after zero_ then backward:", x.grad.item())

# Target value for y (we want to find x such that f(x) = 1.0)
y_target = torch.tensor(1.0)

# Initial guess for x and learning rate
x = torch.tensor(2.5, requires_grad=True)
lr = 1E-3

# Optimization loop
for step in range(100):
    # Evaluate function and compute mean squared error loss
    y = f(x)
    loss = (y - y_target)**2
    loss.backward()

    # Disable gradient tracking for the manual update step
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()

    if step % 5 == 0 or step == 99:
        print(f"step {step:02d}  x={x.item(): .4f}  y={y.item(): .4f}  log(loss)={np.log10(loss.item()): .4f}")



# Define a simple model as a subclass of nn.Module
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define a trainable parameter x
        self.x = nn.Parameter(torch.tensor(2.5))

    # The forward method defines the computation performed at every call
    def forward(self):
        return torch.sin(self.x) * self.x**2


# Initialize the model object and perform a forward pass
toymodel = ToyModel()
y = toymodel()
print(f"y = {y}")

# Access the model parameter x
print(f"model parameter x = {toymodel.x.item()}")

# Inspect the model parameters
for name, p in toymodel.named_parameters():
    print(f"parameter name: {name}, value: {p.item()}, requires_grad: {p.requires_grad}")



y_target = torch.tensor(1.0)

# Use the SGD optimizer from torch.optim instead of manual gradient descent
toyoptimizer = optim.SGD(toymodel.parameters(), lr=1E-3)

# Optimization loop using the optimizer
for step in range(100):
    y = toymodel()
    loss = (y - y_target)**2

    loss.backward()
    toyoptimizer.step()
    toyoptimizer.zero_grad()

    if step % 5 == 0 or step == 99:
        print(f"step {step:02d}  x={toymodel.x.item(): .4f}  y={y.item(): .4f}  log(loss)={np.log10(loss.item()): .4f}")


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor(1.234, requires_grad=True)
print("x device:", x.device)

x = x.to(device)
print("x device after move:", x.device)

y = torch.sin(x) * x**2
y.backward()

print("dy/dx:", x.grad)


# Move a model to the selected device
toymodel = ToyModel()
toymodel = toymodel.to(device)

# Print the model parameters and their devices
for name, p in toymodel.named_parameters():
    print(f"parameter name: {name}, value: {p.item()}, requires_grad: {p.requires_grad}, device: {p.device}")


# Perform a forward pass on the selected device
target = torch.tensor(1.0).to(device)
toyoptimizer = optim.SGD(toymodel.parameters(), lr=1E-3)

# Do one optimization step
toyoptimizer.zero_grad()
y = toymodel()
loss = (y - target)**2
loss.backward()
toyoptimizer.step()

print("x =", toymodel.x.item(), "loss =", loss.item())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Define a random number seed for reproducibility
torch.manual_seed(0)

# Sample synthetic dataset from the function y = x^2 + noise
N = 2048
x = 2.0 * torch.rand(N, 1) - 1.0
y = x**2 + 0.03 * torch.randn(N, 1)

# Train/val split (80% for training and 20% for validation)
n_train = int(0.8 * N)
x_train, x_valid = x[:n_train], x[n_train:]
y_train, y_valid = y[:n_train], y[n_train:]

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_valid, y_valid)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
valid_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

print("train batches:", len(train_loader), "val batches:", len(valid_loader))

# Define the model: a tiny MLP
class ParabolaMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        
        self.net = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

mlp = ParabolaMLP(hidden=32).to(device)
print(mlp)

loss_fn = nn.MSELoss()
mlpopt = optim.Adam(mlp.parameters(), lr=1E-3)


# Make code modular by defining an evaluation function
def evaluate(model, loader):
    model.eval() # Change model to evaluation mode
    total_loss = 0.0
    n = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            
    return total_loss / n

epochs = 100

# Training loop
for epoch in range(1, epochs + 1):
    mlp.train() # Change model to training mode
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        ypred = mlp(xb)
        loss = loss_fn(ypred, yb)

        loss.backward()
        mlpopt.step()
        mlpopt.zero_grad()

    # Evaluate and print losses every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        train_loss = evaluate(mlp, train_loader)
        valid_loss = evaluate(mlp, valid_loader)
        print(f"epoch {epoch:03d} | train log(MSE)={np.log10(train_loss):.5f} | val log(MSE)={np.log10(valid_loss):.5f}")



# Put model in eval mode
mlp.eval()

# Generate a smooth x grid
x_plot = torch.linspace(-1.0, 1.0, 400).unsqueeze(1).to(device)

with torch.no_grad():
    y_pred = mlp(x_plot).cpu()

# Move to CPU for plotting
x_plot = x_plot.cpu()
y_true = x_plot**2

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')

# Training data (subsample for clarity)
ax.scatter(
    x_train[::10].numpy(),
    y_train[::10].numpy(),
    s=10,
    alpha=0.4,
    label="training data",
)

# True function
ax.plot(
    x_plot.numpy(),
    y_true.numpy(),
    linewidth=2,
    label="true y = x²",
)

# Model prediction
ax.plot(
    x_plot.numpy(),
    y_pred.numpy(),
    linewidth=2,
    linestyle="--",
    label="MLP prediction",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.savefig("./plots/output.png", dpi=300, bbox_inches="tight")
plt.close()