# Cell Instance Tracking Pipeline - User Guide

## Overview

This project performs **cell instance tracking** on the **REF_Pos101** microscopy dataset using the **Trackastra** tracking algorithm combined with **SAM2.1** foundation model features.

### Key Features:
- ✅ Load raw images and segmentation masks
- ✅ Run Trackastra tracking in greedy mode
- ✅ Evaluate the results
- ✅ Visualize tracks in Napari (If you want to and have napari installed)

### Goal:
Analyze Trackastra's performance on bacterial cells and understand common tracking errors (e.g., ID switching, merges/splits caused by imperfect segmentation).

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) Napari for visualization

---

## Installation & Setup

### Step 1: Install Required Dependencies

Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install numpy tifffile torch tqdm trackastra traccuracy napari PyQt5
```

**Package breakdown:**
- `numpy` - Array operations
- `tifffile` - TIFF image loading
- `torch` - PyTorch for deep learning models
- `tqdm` - Progress bars
- `trackastra` - Cell tracking algorithm
- `traccuracy` - Tracking evaluation tools
- `napari` - (Optional) Interactive visualization

**Note:** For the visualization script (`view_sam21_result.py`), you'll also need:
```bash
pip install git+https://github.com/bentaculum/divisualisation.git
```

Or install from the local subdirectory if available:
```bash
cd divisualization/divisualisation
pip install -e .
cd ../..
```

---

## Data Requirements

Ensure your data is properly structured in the `data/` directory (you have to create it and put the 4 datafolders in it):

```
data/
├── REF_raw_data101_110 2/
│   └── Pos101/
│       └── aphase/
│           ├── img_000000000.tiff
│           ├── img_000000001.tiff
│           └── ...
└── REF_masks101_110 2/
    └── Pos101/
        └── PreprocessedPhaseMasks/
            ├── MASK_img_000000000.tif
            ├── MASK_img_000000001.tif
            └── ...
```

The pipeline expects:
- **Raw images**: `.tiff` files named `img_*.tiff`
- **Segmentation masks**: `.tif` files named `MASK_img_*.tif`

---

## Usage Workflow

### Step 1: Run Tracking Algorithm

Execute the main tracking script to generate tracking results:

```bash
python run_trackastra_sam21.py
```

**What this does:**
1. Loads raw images from `data/REF_raw_data101_110 2/Pos101/aphase`
2. Loads segmentation masks from `data/REF_masks101_110 2/Pos101/PreprocessedPhaseMasks`
3. Runs Trackastra tracking with SAM2.1 features (greedy mode)
4. Exports results to the `result/` directory in CTC format
5. Generates Napari-compatible track files

**Expected Output:**
```
result/
├── man_track.txt          
├── man_track*.tif           
├── napari_tracks.npy      
├── napari_tracks_graph.npy 
└── tracking_summary.txt   
```

**Processing time:** Depends on dataset size and hardware (GPU/MPS recommended. It is hardcoded to use MPS. If you have CUDA feel free to change it youself).

---

### Step 2: Evaluate Tracking Results

After tracking is complete, evaluate the quality:

```bash
python evaluate.py --result result
```

**What this does:**
- Analyzes tracking results without requiring ground truth
- Computes statistics on track lengths, fragmentation, and consistency
- Identifies potential tracking errors

**Sample Output that I produced on REF_raw_data101_110 2:**
```
==================================================
TRACKING EVALUATION (No Ground Truth)
==================================================

Basic Statistics:
  - Number of frames: 121
  - Total unique track IDs: 290

Track Length Statistics:
  - Average track length: 9.4 frames
  - Median track length: 5.0 frames
  - Min/Max track length: 1/121 frames
  - Tracks < 3 frames (suspicious): 97

Fragmentation Indicators:
  - Avg new tracks per frame: 2.29
  - High values suggest track breaks/fragmentation

Cell Count Consistency:
  - Cell count range: 14 - 36
  - Cell count std dev: 6.50

==================================================
INTERPRETATION:
==================================================
✓ Good: Long average track lengths, low fragmentation
⚠ Bad: Many short tracks, high new tracks per frame
==================================================
```

**Key Metrics:**
- **Track length:** Longer is better (indicates stable tracking)
- **Fragmentation:** Lower new tracks per frame is better
- **Short tracks (<3 frames):** May indicate tracking errors

---

### Step 3: (Optional) Visualize Tracks in Napari

For interactive visualization of tracking results:

```bash
python view_sam21_result.py
```

**What this does:**
1. Loads the raw images and tracking results
2. Opens Napari with the tracks overlaid on images
3. Allows you to navigate through time and inspect individual tracks

**Requirements:**
- Napari must be installed
- The `divisualisation` package must be available
- Display environment (won't work in headless/SSH environments)



## Configuration

To track a different dataset, edit the `DATASET_CONFIG` in `run_trackastra_sam21.py`:

```python
DATASET_CONFIG = {
    'name': 'YourDatasetName',
    'result': 'result',  # Output directory
    'raw_path': 'data/your_raw_path',
    'mask_path': 'data/your_mask_path'
}
```
