#!/usr/bin/env python3
"""
View Trackastra tracking results in napari with tracks layer
"""

import napari
import numpy as np
from pathlib import Path
from skimage import io
from skimage.measure import regionprops
import glob

def load_tracked_masks(result_folder):
    """Load tracked mask files"""
    mask_files = sorted(glob.glob(str(result_folder / "mask*.tif")))
    
    if mask_files:
        masks = [io.imread(f) for f in mask_files]
        stack = np.array(masks)
        print(f"Loaded {len(masks)} tracked masks")
        print(f"Shape: {stack.shape}")
        return stack
    return None

def load_tracking_graph(result_folder):
    """Load tracking graph from res_track.txt (CTC format)"""
    track_file = result_folder / "res_track.txt"
    
    if not track_file.exists():
        return None
    
    # Format: track_id start_frame end_frame parent_id
    tracks = {}
    with open(track_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            track_id = int(parts[0])
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            parent_id = int(parts[3])
            tracks[track_id] = {
                'start': start_frame,
                'end': end_frame,
                'parent': parent_id
            }
    
    return tracks

def extract_centroids_from_masks(masks):
    """Extract cell centroids for each frame"""
    centroids = {}  # {frame: {label_id: (y, x)}}
    
    for t in range(len(masks)):
        centroids[t] = {}
        props = regionprops(masks[t])
        for prop in props:
            centroids[t][prop.label] = prop.centroid
    
    return centroids

def build_tracks_data(masks, tracking_graph):
    """Build napari tracks data from masks and tracking graph"""
    centroids = extract_centroids_from_masks(masks)
    
    tracks_data = []  # Format: [[track_id, t, y, x], ...]
    
    for track_id, info in tracking_graph.items():
        for t in range(info['start'], info['end'] + 1):
            if t in centroids and track_id in centroids[t]:
                y, x = centroids[t][track_id]
                tracks_data.append([track_id, t, y, x])
    
    return np.array(tracks_data)

def main():
    base_dir = Path(__file__).parent
    result_dir = base_dir / "trackastra_results" / "result1"
    
    if not result_dir.exists():
        print(f"❌ Result directory not found: {result_dir}")
        return
    
    # Load tracked masks
    tracked_masks = load_tracked_masks(result_dir)
    
    if tracked_masks is None:
        print("❌ No tracking results found")
        return
    
    # Load tracking graph
    tracking_graph = load_tracking_graph(result_dir)
    
    if tracking_graph is None:
        print("❌ No tracking graph found")
        return
    
    print("\n" + "="*60)
    print("BUILDING TRACKS FROM TRACKING DATA...")
    print("="*60)
    
    # Build tracks for napari
    tracks_data = build_tracks_data(tracked_masks, tracking_graph)
    
    # Count divisions
    divisions = sum(1 for info in tracking_graph.values() if info['parent'] != 0)
    
    print(f"Total tracks: {len(tracking_graph)}")
    print(f"Cell divisions detected: {divisions}")
    print(f"Total track points: {len(tracks_data)}")
    
    # Create viewer
    viewer = napari.Viewer()
    
    # Add tracked masks as labels layer
    viewer.add_labels(tracked_masks, name="Segmentation Masks")
    
    # Add tracks layer
    viewer.add_tracks(tracks_data, name="Cell Tracks", 
                     tail_length=10,  # Show trail of last 10 frames
                     colormap='turbo')
    
    print("\n" + "="*60)
    print("TRACKING RESULTS LOADED")
    print("="*60)
    print(f"Total timepoints: {len(tracked_masks)}")
    print(f"Image dimensions: {tracked_masks.shape[1:]} (Y, X)")
    print(f"\nVisualization:")
    print("  - Tracks layer shows cell trajectories with colored lines")
    print("  - Each track has a tail showing recent movement")
    print("  - Use time slider to see cells moving over time")
    print("="*60 + "\n")
    
    napari.run()

if __name__ == "__main__":
    main()
