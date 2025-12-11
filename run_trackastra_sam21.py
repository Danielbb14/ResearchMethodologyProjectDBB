
from pathlib import Path
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

DATASET_CONFIG = {
    'name': 'REF_Pos101',
    'result': 'result',
    'raw_path': 'data/REF_raw_data101_110 2/Pos101/aphase',
    'mask_path': 'data/REF_masks101_110 2/Pos101/PreprocessedPhaseMasks'
}

def load_image_stack(raw_dir):
    raw_path = Path(raw_dir)
    image_files = sorted(raw_path.glob('img_*.tiff'))
    print(f"Loading {len(image_files)} images from {raw_dir}")
    
    images = []
    for img_file in tqdm(image_files, desc="Loading images"):
        img = tifffile.imread(img_file)
        images.append(img)
    
    imgs = np.stack(images, axis=0)
    print(f"Image stack shape: {imgs.shape}")
    return imgs

def load_mask_stack(mask_dir):
    mask_path = Path(mask_dir)
    mask_files = sorted(mask_path.glob('MASK_img_*.tif'))
    
    print(f"Loading {len(mask_files)} masks from {mask_dir}")
    
    masks = []
    for mask_file in tqdm(mask_files, desc="Loading masks"):
        mask = tifffile.imread(mask_file)
        masks.append(mask)
    
    masks_stack = np.stack(masks, axis=0)
    print(f"Mask stack shape: {masks_stack.shape}")
    return masks_stack

def main():
    print("=" * 80)
    print("Trackastra Et Ultra - SAM2.1 Feature-based Tracking")
    print("=" * 80)
    print(f"\nDataset: {DATASET_CONFIG['name']}")
    print(f"Raw data: {DATASET_CONFIG['raw_path']}")
    print(f"Masks: {DATASET_CONFIG['mask_path']}")
    print(f"Output: {DATASET_CONFIG['result']}")    
    print("\n" + "-" * 80)
    print("Loading data...")
    print("-" * 80)
    imgs = load_image_stack(DATASET_CONFIG['raw_path'])
    masks = load_mask_stack(DATASET_CONFIG['mask_path'])    
    assert imgs.shape[0] == masks.shape[0], \
        f"Number of images ({imgs.shape[0]}) and masks ({masks.shape[0]}) must match"
    
    if imgs.shape[1:] != masks.shape[1:]:
        print(f"\nWarning: Shape mismatch detected!")
        print(f"  Images: {imgs.shape}")
        print(f"  Masks: {masks.shape}")
        print(f"  Cropping masks to match image dimensions...")
        masks = masks[:, :imgs.shape[1], :imgs.shape[2]]
        print(f"  Adjusted mask shape: {masks.shape}")
    
    print("\n" + "-" * 80)
    print("Loading Trackastra model with SAM2.1 features...")
    print("-" * 80)
    
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = Trackastra.from_pretrained("general_2d", device=device)
    print("Model loaded successfully!")
    print("\n" + "-" * 80)
    print("Running tracking with greedy mode...")
    print("-" * 80)

    track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")
    print(f"Tracking complete! Generated {len(track_graph.nodes())} track nodes")
    
    print("\n" + "-" * 80)
    print("Converting to CTC format...")
    print("-" * 80)
    
    output_dir = Path(DATASET_CONFIG['result'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    ctc_tracks, ctc_masks = graph_to_ctc(
        track_graph,
        masks_tracked,
        outdir=str(output_dir)
    )
    print(f"CTC format saved to: {output_dir}")
    
    print("\n" + "-" * 80)
    print("Converting to Napari format...")
    print("-" * 80)
    
    napari_tracks, napari_tracks_graph, napari_graph_meta = graph_to_napari_tracks(track_graph)
    
    np.save(output_dir / 'napari_tracks.npy', napari_tracks)
    np.save(output_dir / 'napari_tracks_graph.npy', napari_tracks_graph)
    print(f"Napari tracks saved to: {output_dir}")
    
    print("\n" + "-" * 80)
    print("Tracking Summary")
    print("-" * 80)
    
    n_frames = imgs.shape[0]
    n_tracks = len(set(napari_tracks[:, 0]))
    n_nodes = len(track_graph.nodes())
    n_edges = len(track_graph.edges())
    
    summary = {
        'dataset': DATASET_CONFIG['name'],
        'model': 'general_2d',
        'n_frames': n_frames,
        'n_tracks': n_tracks,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'image_shape': imgs.shape,
        'device': device
    }
    
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    with open(output_dir / 'tracking_summary.txt', 'w') as f:
        f.write("Trackastra Et Ultra - SAM2.1 Tracking Summary\n")
        f.write("=" * 60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Tracking completed successfully!")
    print("=" * 80)   
   

if __name__ == "__main__":
    main()
