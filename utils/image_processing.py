import numpy as np
import torch
import torch.nn.functional as F

def normalize_image(image):
    """Normalize image to [0,1] range."""
    return (image - image.min()) / (image.max() - image.min())

def undersample_kspace(image, mask):
    """Apply an undersampling mask in k-space."""
    kspace = np.fft.fft2(image)
    kspace = np.fft.fftshift(kspace)
    undersampled_kspace = kspace * mask
    return np.fft.ifftshift(np.fft.ifft2(undersampled_kspace))

def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.var(), img2.var()
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """Load model checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
