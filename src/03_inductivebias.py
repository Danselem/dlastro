import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

class TransitDataset(Dataset):
    def __init__(
        self,
        seq_len: int = 100,
        transit_width: int = 10,
        transit_depth: float = 1.5,
        noise_std: float = 0.3,
        frac_transit: float = 0.5,
        num_samples: int = 5000,
        random_seed: int = 42
    ):
        """Synthetic transit light curve dataset generator."""
        self.data = []
        self.labels = []

        self.seq_len = seq_len
        self.noise_std = noise_std
        self.transit_depth = transit_depth
        self.transit_width = transit_width
        self.frac_transit = frac_transit

        # Set random seed for reproducibility
        self.rng = np.random.default_rng(random_seed)

        # Generate all data on initialization
        for i in range(num_samples):
            flux = self.rng.normal(0, self.noise_std, size=(1, seq_len)).astype(np.float32)

            # Populate transits
            if self.rng.random() < self.frac_transit:
                label = 1
                # Ensure transit is fully contained
                start = self.rng.integers(0, self.seq_len - self.transit_width)
                flux[0, start : start + self.transit_width] -= self.transit_depth
            else:
                label = 0

            self.data.append(torch.from_numpy(flux))
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create a small dataset with 5 samples for visualization
ds = TransitDataset(num_samples=5, frac_transit=0.3, random_seed=42, transit_depth=1.0)

# Show a single item
next(iter(ds))

# Visualize a few examples
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
ds_iter = iter(ds)

for ax in axes.flat:
    data, label = next(ds_iter)

    ax.plot(data.squeeze())
    title = "Transit" if label else "Noise"
    ax.set_title(title)
    ax.set_xlabel("Time step")
    if ax == axes[0]: 
        ax.set_ylabel("Flux")

    ax.axhline(0, color='k', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()

# Make dataset and PyTorch dataloader with lower SNR
dataset = TransitDataset(num_samples=5000, frac_transit=0.5, random_seed=42, noise_std=0.3, transit_depth=0.5)
train_size = int(0.8 * len(dataset))

# Randomly split into 80% training, 20% validation/testing
train_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, len(dataset)-train_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self, seq_len=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # (B, 1, 100) -> (B, 100)
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2), # (B, 1, 100) -> (B, 16, 100)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 16, 50)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),# -> (B, 32, 50)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # -> (B, 32, 25)
            nn.Flatten(),                 # -> (B, 800)
            nn.Linear(32 * 25, 1)
        )

    def forward(self, x):
        return self.net(x)

class SimpleRNN(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        if gated: 
            # Remember/forget gate
            self.rnn = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x is (B, 1, 100); RNN needs transpose (B, Seq, Feature) -> (B, 100, 1)
        x = x.permute(0, 2, 1)
        out, h_n = self.rnn(x)

        # Return final state
        return self.fc(h_n.squeeze(0))


class SimpleTransformer(nn.Module):
    def __init__(self, seq_len=100, include_pos_embed=False):
        super().__init__()
        self.d_model = 32

        # Input embedding
        self.input_proj = nn.Linear(1, self.d_model)

        # Learnable position embeddings (default False)
        if include_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.d_model))
        else:
            self.pos_embed = 0

        # Single-headed attention, encoder only
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

        # Single output
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # x is (B, 1, 100) -> (B, 100, 1)
        x = x.permute(0, 2, 1)

        x = self.input_proj(x) + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_and_evaluate(model, train_loader, test_loader, epochs=20):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    accuracies = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze()
                valid_loss += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        
        # Throttle prints
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1: >2d} | Train Loss: {train_loss/len(train_loader):.4f} | Valid: {valid_loss/len(test_loader):.4f} | Valid Acc: {acc:.2%}")

        accuracies.append(acc)
    return accuracies


def count_trainable_parameters(model):
    """Helper function to count number of trainable params."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("1. MLP")
mlp_model = SimpleMLP()
print(f"    N params: {count_trainable_parameters(mlp_model)}")
mlp_acc = train_and_evaluate(mlp_model, train_loader, test_loader)

print("\n2. CNN")
cnn_model = SimpleCNN()
print(f"    N params: {count_trainable_parameters(cnn_model)}")
cnn_acc = train_and_evaluate(cnn_model, train_loader, test_loader)

print("\n3. RNN")
rnn_model = SimpleRNN()
print(f"    N params: {count_trainable_parameters(rnn_model)}")
rnn_acc = train_and_evaluate(rnn_model, train_loader, test_loader)

print("\n4. Transformer")
transformer_model = SimpleTransformer()
print(f"    N params: {count_trainable_parameters(transformer_model)}")
transformer_acc = train_and_evaluate(transformer_model, train_loader, test_loader)


# Plot the validation accuracy of all models
plt.figure(figsize=(6, 3), dpi=200)

plt.plot(mlp_acc, c="#ffa600", lw=2, ls="-", label="MLP")
plt.plot(cnn_acc, c="#ff6361", lw=2, ls="--", label="CNN")
plt.plot(rnn_acc, c="#bc5090", lw=2, ls="-.", label="RNN")
plt.plot(transformer_acc, c="#58508d", lw=2, ls=":", label="Transformer")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1)
plt.xlim(0, 20)
plt.legend(loc="center right", fontsize=12, framealpha=0)
plt.grid(alpha=0.15)
plt.show()