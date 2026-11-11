# @title Import Pytorch and set device (GPU/CPU)
import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# @title Download catalogs and images
if not os.path.exists('galaxies.csv'):
    print("Downloading catalog...")
    urllib.request.urlretrieve('https://www.dropbox.com/s/e5rfeg71ub8ab0k/galaxies.csv?dl=1', 'galaxies.csv')

if not os.path.exists('images.tar.gz'):
    print("Downloading images...")
    urllib.request.urlretrieve('https://www.dropbox.com/s/ru3cgknmidn0re9/images.tar.gz?dl=1', 'images.tar.gz')

    print("Extracting images...")
    with tarfile.open('images.tar.gz', 'r:gz') as tar:
        tar.extractall()

df = pd.read_csv('galaxies.csv', dtype={'objID': str})
print(f"Dataset contains {len(df)} galaxies.")

print(df.head(3))


# @title View six random galaxies and display their metallicities
n_galaxies = len(df)
rng = np.random.default_rng(seed=42)

fig, axes = plt.subplots(2, 3, figsize=(8, 6), dpi=150)
for i, ax in zip(rng.choice(range(n_galaxies), size=6, replace=False), axes.flat):

    objID = df.objID.iloc[i]
    im = Image.open(f'images/{objID}.jpg')
    ax.imshow(im)
    ax.set_title(objID)
    ax.axis('off')

    Z_gas = df.oh_p50.iloc[i]
    ax.text(0.5, 0.85, f'$Z$ = {Z_gas:.3f}', ha='center', transform=ax.transAxes, size=16, color='white')

fig.subplots_adjust(wspace=0.05, hspace=0.07)

class GalaxyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):

        # store attributes
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

        self.image_ids = self.df['objID'].values
        self.targets = self.df['oh_p50'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # how to index into the dataset and fetch (image, label), where
        #  - the image has shape (3, 160, 160)
        #  - the label is a floating point number
        img_name = os.path.join(self.image_dir, f"{self.image_ids[idx]}.jpg")
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.targets[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


# @title Define $\mathrm D_8$ group with uniform sampling using `torchvision`
def pil_d8_random(img):
    choice = rng.integers(0, 7)
    if choice == 0:
        return img                              # identity
    elif choice == 1:
        return transforms.functional.rotate(img, 90, expand=True)
    elif choice == 2:
        return transforms.functional.rotate(img, 180, expand=True)
    elif choice == 3:
        return transforms.functional.rotate(img, 270, expand=True)
    elif choice == 4:
        return transforms.functional.hflip(img) # horizontal reflection
    elif choice == 5:
        return transforms.functional.vflip(img) # vertical reflection
    elif choice == 6:
        return img.transpose(Image.TRANSPOSE)   # main diagonal
    elif choice == 7:
        return img.transpose(Image.TRANSVERSE)  # anti-diagonal

# Wrap in a torchvision transform
class RandomD8PIL(transforms.RandomApply):
    def __init__(self, p=1.0):
        super().__init__([transforms.Lambda(pil_d8_random)], p=p)



# @title Specify our transforms `train_transforms` and `val_transforms`

# make dummy dataset + dataloader
ds = GalaxyDataset(df, image_dir="images", transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)

# grab a batch of data in order to get image statistics for normalization
xb, yb = next(iter(dl))
img_mean = xb.mean((0,2,3))
img_std = xb.std((0,2,3))


# define training and validation transforms
train_transforms = transforms.Compose([
    RandomD8PIL(),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std)
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std)
])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = GalaxyDataset(train_df, 'images', transform=train_transforms)
val_dataset = GalaxyDataset(val_df, 'images', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)


print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

class GalaxyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # the first convolutional layer, followed by batch normalization,
        # and then the ReLU activation function
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # (B,3,160,160) -> (B,32,80,80)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # second convolutional layer has a similar sequence
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #  -> (B,64,40,40)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # the final fully connected layer is a bit different
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(5),  # creates a 64x5x5 mean pooling layer
            nn.Flatten(),             # unravels the outputs
            nn.Linear(64*5*5, 1)      # just a matrix multiplication
        )

    # this is called whenever we call the model directly
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        return out



model = GalaxyCNN().to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# @title Better weight initialization
def initialize_better(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

model.apply(initialize_better);

learning_rate = 1e-2
weight_decay = 1e-4
epochs = 10

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)


def train_one_epoch(train_loader, model, epoch_index):
    model.train()
    running_loss = 0.0


    batch_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch_index+1}", leave=False)
    for x, y in batch_iterator:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()  # Reset gradients
        p = model(x)           # Forward pass
        p = p.squeeze()        # remove extra dimension
        loss = criterion(p, y) # Calculate loss
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        scheduler.step()       # Update learning rate

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def validate(val_loader, model):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        batch_iterator = tqdm(val_loader, desc="Validating", leave=False)
        for x, y in batch_iterator:
            x, y = x.to(device), y.to(device)
            p = model(x).squeeze()
            loss = criterion(p, y)
            running_loss += loss.item()

            all_preds.extend(p.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return running_loss / len(val_loader), np.array(all_preds), np.array(all_targets)


print("Starting training...")
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = train_one_epoch(train_loader, model, epoch)
    val_loss, _, _ = validate(val_loader, model)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


plt.figure(figsize=(4,3), dpi=150)
plt.plot(np.sqrt(train_losses), label='Train Loss')
plt.plot(np.sqrt(val_losses), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.ylim(0., 0.6)
plt.grid(alpha=0.15)
plt.legend()
plt.show()



# @title Get final predictions
_, preds, targets = validate(val_loader, model)

# Calculate RMSE
rmse = np.sqrt(np.mean((preds - targets)**2))
print(f"Final RMSE: {rmse:.4f} dex")

# Pred vs True plot
plt.figure(figsize=(4,4), dpi=150)
plt.scatter(targets, preds, alpha=0.3, s=10, c='#003f5c', edgecolors='none')
plt.plot([8.0, 9.4], [8.0, 9.4], c='#ff6361', lw=1) # Identity line
plt.xlabel('True Metallicity')
plt.ylabel('CNN Metallicity')
plt.text(0.05, 0.9, f'RMSE = {rmse:.3f}', transform=plt.gca().transAxes, fontsize=16)
plt.grid(True, alpha=0.15)
plt.xlim(8.2, 9.3)
plt.ylim(8.2, 9.3)
plt.show()


from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# load pretrained weights
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)

# repurpose final layer ("classifier") as a regressor
model.classifier = nn.Linear(1280, 1, bias=True)

model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


learning_rate = 3e-4
weight_decay = 0
epochs = 3

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)



print("Training much bigger model...")
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = train_one_epoch(train_loader, model, epoch)
    val_loss, _, _ = validate(val_loader, model)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# @title Again show Predicted vs True plot
_, preds, targets = validate(val_loader, model)
rmse = np.sqrt(np.mean((preds - targets)**2))

# Pred vs True plot
plt.figure(figsize=(4,4), dpi=150)
plt.scatter(targets, preds, alpha=0.3, s=10, c='#003f5c', edgecolors='none')
plt.plot([8.0, 9.4], [8.0, 9.4], c='#ff6361', lw=1) # Identity line
plt.xlabel('True Metallicity')
plt.ylabel('CNN Metallicity')
plt.text(0.05, 0.9, f'RMSE = {rmse:.3f}', transform=plt.gca().transAxes, fontsize=16)
plt.grid(True, alpha=0.15)
plt.xlim(8.2, 9.3)
plt.ylim(8.2, 9.3)
plt.show()