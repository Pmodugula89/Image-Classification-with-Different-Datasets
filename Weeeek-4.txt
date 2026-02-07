import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Transforms
# -------------------------------

mnist_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

color_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Datasets
# -------------------------------

# MNIST (1 channel)
mnist_train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=mnist_transform
)

mnist_test = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=mnist_transform
)

# CIFAR-10 (3 channel)
cifar_train = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=color_transform
)

cifar_test = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=color_transform
)

# Tiny ImageNet (3 channel, 200 classes)
imagenet_train = torchvision.datasets.ImageFolder(
    root="./data/tiny-imagenet-200/train",
    transform=color_transform
)

imagenet_test = torchvision.datasets.ImageFolder(
    root="./data/tiny-imagenet-200/val",
    transform=color_transform
)

# -------------------------------
# DataLoaders
# -------------------------------

batch_size = 32

mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

imagenet_train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True)
imagenet_test_loader = DataLoader(imagenet_test, batch_size=batch_size, shuffle=False)

# -------------------------------
# Visual sanity check
# -------------------------------

sample_image, sample_label = mnist_train[0]
plt.imshow(sample_image.squeeze(), cmap="gray")
plt.title(f"MNIST sample - label {sample_label}")
plt.axis("off")
plt.show()

# -------------------------------
# CNN Model
# -------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Dynamically compute FC size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            dummy_out = self.features(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------------
# Training Function
# -------------------------------

def train_model(model, loader, optimizer, criterion, dataset_name, epochs=5):
    model.train()
    print(f"\nTraining on {dataset_name}")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

# -------------------------------
# Evaluation Function
# -------------------------------

def evaluate_model(model, loader, dataset_name):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"{dataset_name} Test Accuracy: {acc:.4f}")
    return acc

# -------------------------------
# MNIST
# -------------------------------

mnist_model = SimpleCNN(in_channels=1, num_classes=10).to(device)
mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_model(mnist_model, mnist_train_loader, mnist_optimizer, criterion, "MNIST")
mnist_acc = evaluate_model(mnist_model, mnist_test_loader, "MNIST")

# -------------------------------
# CIFAR-10
# -------------------------------

cifar_model = SimpleCNN(in_channels=3, num_classes=10).to(device)
cifar_optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)

train_model(cifar_model, cifar_train_loader, cifar_optimizer, criterion, "CIFAR-10")
cifar_acc = evaluate_model(cifar_model, cifar_test_loader, "CIFAR-10")

# -------------------------------
# Tiny ImageNet
# -------------------------------

imagenet_model = SimpleCNN(in_channels=3, num_classes=200).to(device)
imagenet_optimizer = optim.Adam(imagenet_model.parameters(), lr=0.001)

train_model(imagenet_model, imagenet_train_loader, imagenet_optimizer, criterion, "Tiny ImageNet")
imagenet_acc = evaluate_model(imagenet_model, imagenet_test_loader, "Tiny ImageNet")

# -------------------------------
# Plot Results
# -------------------------------

plt.bar(["MNIST", "CIFAR-10", "Tiny ImageNet"],
        [mnist_acc, cifar_acc, imagenet_acc])
plt.ylabel("Accuracy")
plt.title("CNN Performance Across Datasets")
plt.show()
