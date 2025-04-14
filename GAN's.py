
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
noise_dim = 100
lr = 0.0002
num_epochs = 50
image_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.fc = nn.Linear(noise_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 8, 8)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 8),
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x.view(-1, 1))

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


data_path = "/content/drive/MyDrive/data"

train_dataset = datasets.ImageFolder(root=data_path, transform=transform)


dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True if device == torch.device("cuda") else False)

generator = Generator(noise_dim=noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(20):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)

        real_labels = torch.ones(b_size, 1).to(device)
        fake_labels = torch.zeros(b_size, 1).to(device)

        optimizer_d.zero_grad()

        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(b_size, noise_dim).to(device)
        fake_images = generator(noise)
        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

print("Training completed.")


def denormalize(tensor):
    
    return tensor * 0.5 + 0.5

def show_generated_images(generator, device, num_images=16):
    generator.eval()  
    noise = torch.randn(num_images, noise_dim).to(device)
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    fake_images = denormalize(fake_images)

    grid_size = int(np.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            image = fake_images[idx].permute(1, 2, 0).numpy()  # CHW → HWC
            axes[i, j].imshow(image)
            axes[i, j].axis("off")
            idx += 1
    plt.tight_layout()
    plt.show()

# Call this function to generate and visualize images
show_generated_images(generator, device, num_images=16)

torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

