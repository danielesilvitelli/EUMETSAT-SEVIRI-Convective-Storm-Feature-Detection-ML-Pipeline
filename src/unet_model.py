"""
U-Net Model Architecture Module.
Implements a custom lightweight U-Net for spatial feature detection on multi-channel EO data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet(nn.Module):
    """Custom U-Net architecture designed for binary segmentation of satellite imagery."""
    
    def __init__(self, in_channels=2, out_channels=1):
        super(SimpleUNet, self).__init__()

        # Encoder path (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder path (Upsampling with skip connections)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Input channels: 256 (from up1) + 128 (from skip connection x2) = 384
        self.conv_up1 = DoubleConv(384, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Input channels: 128 (from up2) + 64 (from skip connection x1) = 192
        self.conv_up2 = DoubleConv(192, 128)

        # Final output classification layer
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        bn = self.bottleneck(x3)

        # Decoder forward pass with skip connections
        x = self.up1(bn)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)

        # Output logits mapped to [0, 1] range via Sigmoid activation
        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    # Sanity check test for the model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=2, out_channels=1).to(device)
    
    dummy_input = torch.randn(4, 2, 256, 256).to(device)
    output = model(dummy_input)
    
    print(f"Device in use: {device}")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")