"""
Data Pipeline Module for EUMETSAT SEVIRI Satellite Data.
Handles ingestion, regional cropping, visualization, and PyTorch dataset preparation.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from satpy import Scene
import matplotlib.pyplot as plt

class EumetsatPatchDataset(Dataset):
    """PyTorch Custom Dataset for extracting spatial patches from multi-channel EO tensors."""
    
    def __init__(self, data_tensor, patch_size=256):
        self.data = torch.tensor(data_tensor, dtype=torch.float32)
        self.C, self.H, self.W = self.data.shape
        self.patch_size = patch_size
        
        # Generate patch coordinates with 50% overlap
        self.patches = []
        stride = patch_size // 2
        for y in range(0, self.H - patch_size + 1, stride):
            for x in range(0, self.W - patch_size + 1, stride):
                self.patches.append((y, x))
                
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        y, x = self.patches[idx]
        patch = self.data[:, y:y+self.patch_size, x:x+self.patch_size]
        return patch


def normalize_array(arr):
    """Apply Min-Max normalization to scale array values between [0, 1]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def run_data_pipeline(file_path_str, output_img_path):
    """Execute the end-to-end EO data processing and PyTorch DataLoader generation."""
    file_path = Path(file_path_str)
    assert file_path.exists(), f"File not found: {file_path}"
    
    print(f"Loading file: {file_path.name} ({file_path.stat().st_size / 1e6:.1f} MB)")

    # 1. Initialize Satpy Scene from native format
    scn = Scene(reader="seviri_l1b_native", filenames=[str(file_path)])
    
    # 2. Load required infrared and water vapor channels
    channels_to_load = ["IR_108", "WV_062"]
    scn.load(channels_to_load)

    # 3. Resample and crop scene to European sector ('eurol')
    scn_cropped = scn.resample('eurol')
    print(f"Resampling successful. Cropped shape (IR_108): {scn_cropped['IR_108'].shape}")

    # 4. Save preprocessed quicklook image with coastlines
    output_img = Path(output_img_path)
    output_img.parent.mkdir(parents=True, exist_ok=True)
    
    scn_cropped.save_dataset(
        "IR_108",
        filename=str(output_img),
        overlay_coastlines=True,
        enhance=True
    )
    print(f"Preprocessed image saved to: {output_img}")

    # 5. Extract DataArrays and convert to NumPy arrays
    ir108_arr = np.nan_to_num(scn_cropped["IR_108"].values, nan=0.0)
    wv062_arr = np.nan_to_num(scn_cropped["WV_062"].values, nan=0.0)

    # 6. Normalize channels and stack into a multi-channel tensor (C, H, W)
    ir108_norm = normalize_array(ir108_arr)
    wv062_norm = normalize_array(wv062_arr)
    combined_data = np.stack([ir108_norm, wv062_norm], axis=0).astype(np.float32)
    
    print(f"Combined multi-channel tensor shape: {combined_data.shape}")

    # 7. Initialize PyTorch Dataset and DataLoader
    dataset = EumetsatPatchDataset(combined_data, patch_size=256)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset successfully created! Total patches extracted: {len(dataset)}")

    # 8. Quick test batch validation
    for batch in dataloader:
        print(f"DataLoader batch shape: {batch.shape}")
        break
        
    return dataloader


if __name__ == "__main__":
    # Example paths for local execution or testing
    RAW_FILE = "data/raw/MSG3-SEVI-MSG15-0100-NA-20261024141243.420000000Z-NA.nat"
    PROCESSED_IMG = "data/processed/europe_ir108.png"
    
    # Run pipeline if executed directly
    # dataloader = run_data_pipeline(RAW_FILE, PROCESSED_IMG)