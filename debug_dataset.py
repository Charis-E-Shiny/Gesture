import json
from pathlib import Path

# Load metadata
with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Check split
split = 'train'
print(f'Available splits: {list(metadata["splits"].keys())}')
print(f'Train videos count in metadata: {metadata["splits"][split]["count"]}')

# Check first video entry
video_entry = metadata['splits'][split]['videos'][0]
print(f'First video entry: {video_entry}')

# Build path
data_dir = Path('data/train')
filename = video_entry['filename']
video_path = data_dir / filename

print(f'Constructed path: {video_path}')
print(f'Exists: {video_path.exists()}')

# Now test the _load_from_metadata function
from data_extractor import LandmarkDataset

print("\n--- Testing LandmarkDataset ---")
dataset = LandmarkDataset(data_dir='data/train', split='train')
print(f'Dataset samples loaded: {len(dataset.samples)}')
if len(dataset.samples) > 0:
    print(f'First sample: {dataset.samples[0]}')
