import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# --- 0. Hardware Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. The Authentic Skein-Convolutional Layer ---
class AuthenticSkeinConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(AuthenticSkeinConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # We only define ONE set of weights for the 'positive crossing'
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Conway's Z parameter (the smoothing coefficient)
        self.z = nn.Parameter(torch.tensor([1.0]))

        # 1x1 conv to make sure L0 (smoothed) has the right number of channels
        self.smooth_projector = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # L+: The primary learnable filter (Positive Crossing)
        lp = F.conv2d(x, self.weight, bias=self.bias, padding=1)

        # L-: The Negative Crossing (Mirrored Weights)
        # We flip the height and width dims of the kernel to represent the opposite crossing
        weight_minus = torch.flip(self.weight, dims=[2, 3])
        lm = F.conv2d(x, weight_minus, bias=self.bias, padding=1)

        # L0: The "Smoothed" state (Topological reconnecting)
        # We use a Blur/Average pool to simulate the smoothing of the knot
        l0_raw = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        l0 = self.smooth_projector(l0_raw)

        # Apply Conway Identity: L+ - L- - (z * L0)
        # We wrap them in activation to maintain non-linearity
        return self.act(lp) - self.act(lm) - (self.z * self.act(l0))

# --- 2. The Topological CNN Network ---
class SkeinCNN(nn.Module):
    def __init__(self):
        super(SkeinCNN, self).__init__()
        self.skein1 = AuthenticSkeinConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.skein2 = AuthenticSkeinConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.skein1(x))
        x = self.pool2(self.skein2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 3. Data Preparation ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=128, shuffle=False)

# --- 4. Training ---
model = SkeinCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

print(f"Training 'Authentic' Skein-CNN on {device}...")

for epoch in range(5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Quick Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()

    print(f"Epoch {epoch+1} | Accuracy: {100. * correct / len(test_loader.dataset):.2f}% | Z-val: {model.skein1.z.item():.3f}")

# --- 5. Conclusion ---
print("\nSuccess. The model is now using constrained weights to simulate topological crossings.")

# --- 6. Real Dataset Inference Test & Evaluation ---
print("\n" + "="*50)
print("--- Authentic Skein-CNN Inference Test ---")
print("="*50)

model.eval()

# Grab a single batch of test data
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Let's test on the first 15 images of this unseen batch
num_samples = 15
test_images = images[:num_samples]
test_targets = labels[:num_samples]

# Run inference without tracking gradients
with torch.no_grad():
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)

# 1. Print the learned Topological/Smoothing weights
print(f"Final Learned 'Z' Smoothing Weight (Layer 1): {model.skein1.z.item():.4f}")
print(f"Final Learned 'Z' Smoothing Weight (Layer 2): {model.skein2.z.item():.4f}\n")

# 2. Print the prediction table
print(f"{'Sample':<8} | {'Target':<8} | {'Predicted':<10} | {'Result'}")
print("-" * 45)

correct_count = 0
for i in range(num_samples):
    target = test_targets[i].item()
    pred = predictions[i].item()

    match = "✅ Match" if target == pred else "❌ Miss"
    if target == pred:
        correct_count += 1

    print(f"{i+1:<8} | {target:<8} | {pred:<10} | {match}")

print("-" * 45)
print(f"Inference Accuracy on this subset: {correct_count}/{num_samples} ({(correct_count/num_samples)*100:.1f}%)")
print("="*50)
