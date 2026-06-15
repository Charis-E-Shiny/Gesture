import json
from pathlib import Path

# Load metadata
with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Manual load from metadata
data_dir = Path('data/train')
split = 'train'
metadata_file = 'data/metadata.json'

samples = []
if split in metadata['splits']:
    print(f"Split {split} found in metadata")
    videos = metadata['splits'][split]['videos']
    print(f"Total videos in split: {len(videos)}")
    
    for i, video_entry in enumerate(videos):
        filename = video_entry['filename']
        gloss = video_entry['gloss']
        
        # Construct video path
        video_path = data_dir / filename
        
        print(f"  Video {i}: {filename} -> {video_path} (exists: {video_path.exists()})")
        
        if video_path.exists():
            samples.append((str(video_path), gloss))
        
        if i >= 5:
            print("  ...")
            break

print(f"\nLoaded {len(samples)} samples (first 5 shown)")

# Now test with LandmarkDataset
print("\n--- Testing LandmarkDataset with metadata_file ---")
from data_extractor import LandmarkDataset

dataset = LandmarkDataset(data_dir='data/train', split='train', metadata_file='data/metadata.json')
print(f'Dataset samples loaded: {len(dataset.samples)}')
if len(dataset.samples) > 0:
    print(f'First 3 samples: {dataset.samples[:3]}')
