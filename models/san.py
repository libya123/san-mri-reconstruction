import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(attn))

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = torch.mean(x, dim=[2, 3])
        max_out, _ = torch.max(x, dim=[2, 3])
        attn = self.fc(avg_out) + self.fc(max_out)
        return x * attn.unsqueeze(2).unsqueeze(3)

class MultiResGenerator(nn.Module):
    def __init__(self):
        super(MultiResGenerator, self).__init__()
        self.low_res = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        self.mid_res = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            SpatialAttention(),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.high_res = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            ChannelAttention(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        low = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        mid = x
        high = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        return self.low_res(low) + self.mid_res(mid) + self.high_res(high)

class FeatureMatchingDiscriminator(nn.Module):
    def __init__(self):
        super(FeatureMatchingDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 16 * 16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return torch.sigmoid(self.fc(x.view(x.shape[0], -1)))
