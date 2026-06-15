import json

# Verify datasets created
print('DATASET VERIFICATION')
print('=' * 60)

with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Metadata loaded: {metadata["dataset"]} v{metadata["version"]}')
print(f'Total videos: {metadata["total_videos"]}')
print(f'Total glosses: {metadata["total_glosses"]}')
print()
print('Splits:')
for split in ['train', 'val', 'test']:
    count = metadata['splits'][split]['count']
    print(f'  {split.upper()}: {count} videos')

print()
print('Sample glosses:')
vocab = metadata['gloss_vocab']
for i in range(5):
    print(f'  {i}: {vocab[str(i)]}')

print()
print('Dataset ready for training!')
print('=' * 60)
