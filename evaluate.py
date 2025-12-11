"""Simple tracking evaluation without ground truth"""
import numpy as np
from pathlib import Path
from traccuracy. loaders import load_ctc_data

def evaluate_tracks(result_path):
    track_file = result_path / "res_track.txt"
    if not track_file.exists():
        track_file = result_path / "man_track.txt"
    
    pred = load_ctc_data(
        str(result_path),
        str(track_file),
        run_checks=False,
        name="prediction",
    )
    
    masks = pred. segmentation
    graph = pred.graph
    
    print("=" * 50)
    print("TRACKING EVALUATION (No Ground Truth)")
    print("=" * 50)
    
    num_frames = masks.shape[0]
    all_track_ids = set()
    for t in range(num_frames):
        all_track_ids.update(np.unique(masks[t]))
    all_track_ids. discard(0)
    
    print(f"\nBasic Statistics:")
    print(f"  - Number of frames: {num_frames}")
    print(f"  - Total unique track IDs: {len(all_track_ids)}")
    
    track_lengths = {}
    for track_id in all_track_ids:
        length = sum(1 for t in range(num_frames) if track_id in masks[t])
        track_lengths[track_id] = length
    
    lengths = list(track_lengths.values())
    print(f"\nTrack Length Statistics:")
    print(f"  - Average track length: {np.mean(lengths):.1f} frames")
    print(f"  - Median track length: {np.median(lengths):.1f} frames")
    print(f"  - Min/Max track length: {min(lengths)}/{max(lengths)} frames")
    print(f"  - Tracks < 3 frames (suspicious): {sum(1 for l in lengths if l < 3)}")
    
    new_tracks = []
    prev_ids = set()
    for t in range(num_frames):
        curr_ids = set(np.unique(masks[t])) - {0}
        new_tracks.append(len(curr_ids - prev_ids))
        prev_ids = curr_ids
    
    print(f"\nFragmentation Indicators:")
    print(f"  - Avg new tracks per frame: {np.mean(new_tracks[1:]):.2f}")
    print(f"  - High values suggest track breaks/fragmentation")
    
    counts = [len(np.unique(masks[t])) - 1 for t in range(num_frames)]
    print(f"\nCell Count Consistency:")
    print(f"  - Cell count range: {min(counts)} - {max(counts)}")
    print(f"  - Cell count std dev: {np.std(counts):.2f}")
    
    print("\n" + "=" * 50)
    print("INTERPRETATION:")
    print("=" * 50)
    print("✓ Good: Long average track lengths, low fragmentation")
    print("⚠ Bad: Many short tracks, high new tracks per frame")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, required=True)
    args = parser.parse_args()
    evaluate_tracks(Path(args.result))