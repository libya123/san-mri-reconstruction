import torch
import torch.optim as optim
import torch.nn as nn
from dataset.fastmri_loader import get_fastmri_loader
from models.san import MultiResGenerator, FeatureMatchingDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_fastmri_loader("path_to_dataset", batch_size=8)

G = MultiResGenerator().to(device)
D = FeatureMatchingDiscriminator().to(device)

adversarial_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

num_epochs = 10
for epoch in range(num_epochs):
    for real_images, _ in train_loader:
        real_images = real_images.to(device)
        
        optimizer_D.zero_grad()
        fake_images = G(real_images)
        loss_D_real = adversarial_loss(D(real_images), torch.ones_like(D(real_images)))
        loss_D_fake = adversarial_loss(D(fake_images.detach()), torch.zeros_like(D(fake_images)))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        loss_G_adversarial = adversarial_loss(D(fake_images), torch.ones_like(D(fake_images)))
        loss_G_reconstruction = mse_loss(fake_images, real_images)
        loss_G = loss_G_adversarial + 0.1 * loss_G_reconstruction
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_D.item()} Loss G: {loss_G.item()}")
