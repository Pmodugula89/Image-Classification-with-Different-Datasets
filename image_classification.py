import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define dataset transformations (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust for dataset normalization
])

# Load MNIST (grayscale dataset)
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load CIFAR-10 (color dataset)
cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Load a subset of ImageNet (pre-downloaded tiny version)
imagenet_trainset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
imagenet_testset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)

# Create data loaders
batch_size = 32
mnist_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
cifar_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
imagenet_loader = torch.utils.data.DataLoader(imagenet_trainset, batch_size=batch_size, shuffle=True)

# Display a sample image
sample_image, sample_label = mnist_trainset[0]
plt.imshow(sample_image.squeeze(), cmap='gray')  # Convert tensor to image
plt.title(f"Sample Image - MNIST Class {sample_label}")
plt.axis("off")
plt.show()
