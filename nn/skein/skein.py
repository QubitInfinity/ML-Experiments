import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# --- 0. Hardware Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# --- 1. The Skein-Convolutional Layer ---
class SkeinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SkeinConv2d, self).__init__()

        # The three topological states are now spatial filters
        self.conv_plus = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_minus = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_smooth = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        self.z = nn.Parameter(torch.tensor([1.5]))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        lp = self.act(self.conv_plus(x))
        lm = self.act(self.conv_minus(x))
        l0 = self.act(self.conv_smooth(x))

        # Conway Spatial Identity: L+ - L- - (z * L0)
        return lp - lm - (self.z * l0)

# --- 2. The Topological CNN Network ---
class SkeinCNN(nn.Module):
    def __init__(self):
        super(SkeinCNN, self).__init__()

        # Block 1: Input is 1 channel (grayscale) -> Outputs 16 feature maps
        self.skein1 = SkeinConv2d(in_channels=1, out_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 28x28 to 14x14

        # Block 2: Takes 16 feature maps -> Outputs 32 feature maps
        self.skein2 = SkeinConv2d(in_channels=16, out_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks 14x14 to 7x7

        # Final Classification: 32 channels * 7 height * 7 width = 1568
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x shape: [Batch, 1, 28, 28] (Standard 2D image)
        x = self.pool1(self.skein1(x))
        x = self.pool2(self.skein2(x))

        # Flatten the 2D feature maps into a 1D vector for the final guessing layer
        x = x.view(x.size(0), -1)

        return self.fc(x)

# --- 3. Data Preparation ---
print("Loading 28x28 Standard MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# --- 4. Training Loop ---
model = SkeinCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training Skein-CNN Topological Vision Model...")
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device) # Move data to GPU/CPU

        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            test_out = model(batch_X)
            predictions = test_out.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2} | Avg Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

# --- 5. Real Dataset Inference Test (100 Samples) ---
print("\n--- Skein-CNN Topological MNIST Test (100 Real Samples) ---")
model.eval()

with torch.no_grad():
    test_loader_100 = DataLoader(test_dataset, batch_size=100, shuffle=True)
    test_images, test_targets = next(iter(test_loader_100))
    test_images, test_targets = test_images.to(device), test_targets.to(device)

    predictions = model(test_images).argmax(dim=1)

    correct = (predictions == test_targets).sum().item()
    accuracy = (correct / 100.0) * 100

print(f"Accuracy over 100 unseen samples: {accuracy:.2f}%")
print(f"Final Learned Conway Weight (Z) Layer 1: {model.skein1.z.item():.4f}")
print(f"Final Learned Conway Weight (Z) Layer 2: {model.skein2.z.item():.4f}\n")

print("Snapshot of first 10 predictions:")
for i in range(10):
    target = test_targets[i].item()
    pred = predictions[i].item()
    match = "✅" if target == pred else "❌"
    print(f"Sample {i+1:2d} | Target: {target} | Predicted: {pred} {match}")
