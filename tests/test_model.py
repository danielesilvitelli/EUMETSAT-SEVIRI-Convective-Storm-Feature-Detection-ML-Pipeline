"""
Test and Inference Module for EUMETSAT SEVIRI Convective Storm Detection.
Evaluates the trained U-Net model on a selected patch and generates the visual comparison plot.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.unet_model import SimpleUNet
from src.data_pipeline import run_data_pipeline


def evaluate_and_plot(model, dataloader, device, output_img_path="docs/results_preview.png"):
    """Perform inference on the most cloudy patch and save the 3-panel comparison plot."""
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_gpu = batch.to(device)

            # Select the patch with the lowest mean IR temperature (intense clouds/storms)
            ir_channels = batch_gpu[:, 0, :, :]
            means = ir_channels.mean(dim=(1, 2))
            best_idx = torch.argmin(means).item()

            selected_patch = batch_gpu[best_idx : best_idx + 1]
            preds = model(selected_patch)

            ir_img = selected_patch[0, 0].cpu().numpy()

            # Generate local physical ground truth mask based on temperature percentiles
            local_thresh = np.percentile(ir_img, 15)
            target_mask = (ir_img <= local_thresh).astype(np.float32)

            pred_mask = preds[0, 0].cpu().detach().numpy()
            break

    # Create the 3-panel comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Raw Infrared Input Patch
    im0 = axes[0].imshow(ir_img, cmap="inferno")
    axes[0].set_title("1. SEVIRI IR 10.8 µm (Input Patch)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Physical Ground Truth Mask
    im1 = axes[1].imshow(target_mask, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("2. Physical Ground Truth (Convective Cores)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: U-Net Model Prediction Output
    im2 = axes[2].imshow(pred_mask, cmap="Purples", vmin=0, vmax=1)
    axes[2].set_title("3. U-Net Feature Detection (Output)", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("EUMETSAT SEVIRI - Convective Storm Feature Detection Pipeline", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save output image to docs/
    output_path = Path(output_img_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Final results preview successfully saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Set execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    RAW_FILE = "data/raw/MSG3-SEVI-MSG15-0100-NA-20261024141243.420000000Z-NA.nat"
    PROCESSED_IMG = "data/processed/europe_ir108.png"
    WEIGHTS_PATH = "data/processed/unet_seviri_weights.pth"
    OUTPUT_PREVIEW = "docs/results_preview.png"

    # 1. Load data pipeline
    dataloader = run_data_pipeline(RAW_FILE, PROCESSED_IMG)

    # 2. Initialize model and load trained weights (if available, otherwise random initialization for testing)
    model = SimpleUNet(in_channels=2, out_channels=1).to(device)
    if Path(WEIGHTS_PATH).exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("Trained model weights loaded successfully.")
    else:
        print("Warning: Weights file not found. Running with uninitialized model weights.")

    # 3. Run evaluation and generate preview plot
    evaluate_and_plot(model, dataloader, device, OUTPUT_PREVIEW)