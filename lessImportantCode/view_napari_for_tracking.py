#!/usr/bin/env python3
"""
Napari viewer optimized for Trackastra cell tracking
Ensures images and masks have matching dimensions (Time, Y, X)
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
        stack = np.array(images)
        
        # Ensure 3D: (Time, Y, X) - squeeze out extra dimensions if present
        while stack.ndim > 3:
            # If there's a channel dimension of size 1, remove it
            if stack.shape[1] == 1:
                stack = stack.squeeze(axis=1)
            # If there are multiple channels, take the first one (or you could do max projection)
            elif stack.ndim == 4:
                print(f"Warning: Multi-channel image detected, taking first channel. Shape was {stack.shape}")
                stack = stack[:, 0, :, :]  # Take first channel
            else:
                break
        
        return stack
    return None

def load_mask_stack(mask_folder):
    """Load a stack of segmentation mask images"""
    mask_files = sorted(glob.glob(str(mask_folder / "MASK_*.tif")))
    
    if mask_files:
        masks = [io.imread(f) for f in mask_files]
        stack = np.array(masks)
        
        # Ensure 3D: (Time, Y, X) - squeeze out extra dimensions if present
        while stack.ndim > 3:
            if stack.shape[1] == 1:
                stack = stack.squeeze(axis=1)
            else:
                stack = stack[:, 0, :, :]
                
        return stack
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
    
    print("\n" + "="*60)
    print("LOADING DATA FOR TRACKASTRA TRACKING")
    print("="*60)
    
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
                    # Load corresponding masks
                    mask_folder = ref_masks / pos_name / "PreprocessedPhaseMasks"
                    if mask_folder.exists():
                        mask_stack = load_mask_stack(mask_folder)
                        if mask_stack is not None:
                            # Verify dimensions match
                            if image_stack.shape != mask_stack.shape:
                                print(f"\n⚠️  WARNING: Dimension mismatch for {pos_name}!")
                                print(f"   Image shape: {image_stack.shape}")
                                print(f"   Mask shape:  {mask_stack.shape}")
                                
                                # Try to fix by matching time dimension
                                min_time = min(image_stack.shape[0], mask_stack.shape[0])
                                image_stack = image_stack[:min_time]
                                mask_stack = mask_stack[:min_time]
                                
                                # Check spatial dimensions
                                if image_stack.shape[1:] != mask_stack.shape[1:]:
                                    # Try to crop mask to match image dimensions
                                    img_h, img_w = image_stack.shape[1:]
                                    mask_h, mask_w = mask_stack.shape[1:]
                                    
                                    if mask_h >= img_h and mask_w >= img_w:
                                        # Crop mask from top-left to match image
                                        mask_stack = mask_stack[:, :img_h, :img_w]
                                        print(f"   ✓ Fixed by cropping mask to {image_stack.shape}")
                                    else:
                                        print(f"   ❌ Cannot fix: mask smaller than image, skipping {pos_name}")
                                        continue
                                else:
                                    print(f"   ✓ Fixed by trimming to {min_time} timepoints")
                            
                            # Add to viewer
                            viewer.add_image(image_stack, name=f"REF_{pos_name}_raw", 
                                           colormap='gray', contrast_limits=[image_stack.min(), image_stack.max()])
                            viewer.add_labels(mask_stack, name=f"REF_{pos_name}_masks")
                            
                            print(f"\n✓ Loaded {pos_name}:")
                            print(f"  - Shape: {image_stack.shape} (Time, Y, X)")
                            print(f"  - {len(image_stack)} timepoints")
                            print(f"  - Image and mask dimensions MATCH ✓")
    
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
                    # Load corresponding masks
                    mask_folder = rif_masks / pos_name / "PreprocessedPhaseMasks"
                    if mask_folder.exists():
                        mask_stack = load_mask_stack(mask_folder)
                        if mask_stack is not None:
                            # Verify dimensions match
                            if image_stack.shape != mask_stack.shape:
                                print(f"\n⚠️  WARNING: Dimension mismatch for {pos_name}!")
                                print(f"   Image shape: {image_stack.shape}")
                                print(f"   Mask shape:  {mask_stack.shape}")
                                
                                # Try to fix by matching time dimension
                                min_time = min(image_stack.shape[0], mask_stack.shape[0])
                                image_stack = image_stack[:min_time]
                                mask_stack = mask_stack[:min_time]
                                
                                # Check spatial dimensions
                                if image_stack.shape[1:] != mask_stack.shape[1:]:
                                    # Try to crop mask to match image dimensions
                                    img_h, img_w = image_stack.shape[1:]
                                    mask_h, mask_w = mask_stack.shape[1:]
                                    
                                    if mask_h >= img_h and mask_w >= img_w:
                                        # Crop mask from center or top-left
                                        # Assuming padding might be on edges, crop to match
                                        mask_stack = mask_stack[:, :img_h, :img_w]
                                        print(f"   ✓ Fixed by cropping mask to {image_stack.shape}")
                                    else:
                                        print(f"   ❌ Cannot fix: mask smaller than image, skipping {pos_name}")
                                        continue
                                else:
                                    print(f"   ✓ Fixed by trimming to {min_time} timepoints")
                            
                            # Add to viewer
                            viewer.add_image(image_stack, name=f"RIF10_{pos_name}_raw",
                                           colormap='gray', contrast_limits=[image_stack.min(), image_stack.max()])
                            viewer.add_labels(mask_stack, name=f"RIF10_{pos_name}_masks")
                            
                            print(f"\n✓ Loaded {pos_name}:")
                            print(f"  - Shape: {image_stack.shape} (Time, Y, X)")
                            print(f"  - {len(image_stack)} timepoints")
                            print(f"  - Image and mask dimensions MATCH ✓")
    
    print("\n" + "="*60)
    print("READY FOR TRACKASTRA TRACKING")
    print("="*60)
    print("\nTo track cells in napari:")
    print("1. Go to: Plugins → trackastra → Track")
    print("2. Select an image layer (e.g., REF_Pos101_raw)")
    print("3. Select matching labels layer (e.g., REF_Pos101_masks)")
    print("4. Choose model: 'general_2d' (recommended)")
    print("5. Click 'Run'")
    print("\nDimensions are verified to match - tracking should work! ✓")
    print("="*60 + "\n")
    
    napari.run()

if __name__ == "__main__":
    main()
