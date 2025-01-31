import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.fastmri_loader import get_fastmri_loader
from models.san import MultiResGenerator
from utils.image_processing import compute_psnr, compute_ssim, normalize_image

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = MultiResGenerator().to(device)
G.load_state_dict(torch.load("checkpoint.pth", map_location=device))
G.eval()

# Load test dataset
test_loader = get_fastmri_loader("path_to_dataset", batch_size=1, train=False)

# Evaluate model
for real_images, _ in test_loader:
    real_images = real_images.to(device)

    # Reconstruct images
    with torch.no_grad():
        reconstructed_images = G(real_images)

    # Convert to numpy
    real_images_np = real_images.cpu().numpy().squeeze()
    reconstructed_images_np = reconstructed_images.cpu().numpy().squeeze()

    # Compute Metrics
    psnr_value = compute_psnr(real_images_np, reconstructed_images_np)
    ssim_value = compute_ssim(real_images_np, reconstructed_images_np)

    # Display Results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_images_np, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(reconstructed_images_np, cmap='gray')
    axes[1].set_title(f"Reconstructed (PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f})")
    plt.show()

    break  # Only process one sample
