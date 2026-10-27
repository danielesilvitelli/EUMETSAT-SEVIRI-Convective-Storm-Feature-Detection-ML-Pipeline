import torch
from src.unet_model import SimpleUNet

def test_unet_output_shape():
    """Test that the U-Net model returns the correct tensor shape."""
    model = SimpleUNet(in_channels=2, out_channels=1)
    model.eval()
    
    dummy_input = torch.randn(2, 2, 256, 256) # Batch size 2, 2 channels, 256x256
    output = model(dummy_input)
    
    # Expected output shape: [Batch_Size, Out_Channels, Height, Width] => [2, 1, 256, 256]
    assert output.shape == (2, 1, 256, 256), f"Unexpected output shape: {output.shape}"