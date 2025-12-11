#!/usr/bin/env python3
"""
Napari viewer for microscopy data with segmentation masks
Loads time-series image sequences and corresponding Omnipose masks
"""

import napari
import numpy as np
from pathlib import Path
from skimage import io
import glob

def load_image_stack(image_folder, pattern="*.tiff"):
    """Load a stack of TIFF images from a folder"""
    image_files = sorted(glob.glob(str(image_folder / pattern)))
    if not image_files and pattern == "*.tiff":
        image_files = sorted(glob.glob(str(image_folder / "*.tif")))
    
    if image_files:
        images = [io.imread(f) for f in image_files]
        return np.array(images)
    return None

def load_mask_stack(mask_folder):
    """Load a stack of segmentation mask images"""
    mask_files = sorted(glob.glob(str(mask_folder / "MASK_*.tif")))
    
    if mask_files:
        masks = [io.imread(f) for f in mask_files]
        return np.array(masks)
    return None



def main():
    # Base directory
    base_dir = Path(__file__).parent
    
    # Data folders
    ref_raw = base_dir / "REF_raw_data101_110 2"
    ref_masks = base_dir / "REF_masks101_110 2"
    rif_raw = base_dir / "RIF10_raw_data201_210 2"
    rif_masks = base_dir / "RIF10_masks201_210 2"
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Load REF data (untreated cells, Positions 101-110)
    if ref_raw.exists() and ref_masks.exists():
        positions = sorted([p for p in ref_raw.iterdir() if p.is_dir() and p.name.startswith("Pos")])
        
        for pos in positions[:2]:  # Load first 2 positions as example
            pos_name = pos.name
            
            # Load raw images
            aphase_folder = pos / "aphase"
            if aphase_folder.exists():
                image_stack = load_image_stack(aphase_folder)
                if image_stack is not None:
                    viewer.add_image(image_stack, name=f"REF_{pos_name}_raw", 
                                   colormap='gray', contrast_limits=[image_stack.min(), image_stack.max()])
                    
                    # Load corresponding masks
                    mask_folder = ref_masks / pos_name / "PreprocessedPhaseMasks"
                    if mask_folder.exists():
                        mask_stack = load_mask_stack(mask_folder)
                        if mask_stack is not None:
                            viewer.add_labels(mask_stack, name=f"REF_{pos_name}_masks")
                            print(f"Loaded {pos_name}: {len(image_stack)} raw images and {len(mask_stack)} masks")
    
    # Load RIF10 data (treated cells, Positions 201-210)
    if rif_raw.exists() and rif_masks.exists():
        positions = sorted([p for p in rif_raw.iterdir() if p.is_dir() and p.name.startswith("Pos")])
        
        for pos in positions[:2]:  # Load first 2 positions as example
            pos_name = pos.name
            
            # Load raw images
            aphase_folder = pos / "aphase"
            if aphase_folder.exists():
                image_stack = load_image_stack(aphase_folder)
                if image_stack is not None:
                    viewer.add_image(image_stack, name=f"RIF10_{pos_name}_raw",
                                   colormap='gray', contrast_limits=[image_stack.min(), image_stack.max()])
                    
                    # Load corresponding masks
                    mask_folder = rif_masks / pos_name / "PreprocessedPhaseMasks"
                    if mask_folder.exists():
                        mask_stack = load_mask_stack(mask_folder)
                        if mask_stack is not None:
                            viewer.add_labels(mask_stack, name=f"RIF10_{pos_name}_masks")
                            print(f"Loaded {pos_name}: {len(image_stack)} raw images and {len(mask_stack)} masks")
    
    print("\nNapari viewer started with time-series data and segmentation masks.")
    print("Use the slider to navigate through time points.")
    print("Toggle layer visibility with the eye icon.")
    print("Close the window to exit.")
    napari.run()

if __name__ == "__main__":
    main()
