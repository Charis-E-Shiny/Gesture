#!/usr/bin/env python3
"""
Organize WLASL videos into train/val/test splits with proper metadata mapping
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

def organize_wlasl_data():
    """Main function to organize WLASL data"""
    
    print("=" * 80)
    print("ORGANIZING WLASL DATA FOR TRAINING")
    print("=" * 80)
    
    # Paths
    wlasl_json = Path("WLASL_temp/start_kit/WLASL_v0.3.json")
    videos_src = Path("WLASL_temp/start_kit/raw_videos")
    data_dir = Path("data")
    
    # Create data directories
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {d}")
    
    print("\n[STEP 1] Loading WLASL metadata...")
    with open(wlasl_json, 'r') as f:
        wlasl_data = json.load(f)
    
    print(f"  Total glosses: {len(wlasl_data)}")
    
    # Build mapping: video_id -> gloss info
    video_to_gloss = {}
    gloss_vocab = {}
    gloss_id = 0
    gloss_to_instances = defaultdict(list)
    
    print("\n[STEP 2] Building video-to-gloss mapping...")
    for gloss_entry in wlasl_data:
        gloss_name = gloss_entry['gloss']
        gloss_vocab[gloss_id] = gloss_name
        
        for instance in gloss_entry['instances']:
            # Video ID is the video filename (like 00623, 01383, etc.)
            video_id = instance.get('video_id')
            if video_id is None:
                # Fallback: construct from instance_id
                video_id = f"{instance['instance_id']:05d}"
            
            video_to_gloss[video_id] = {
                'gloss_id': gloss_id,
                'gloss_name': gloss_name,
                'instance_id': instance['instance_id'],
                'bbox': instance.get('bbox'),
                'frame_start': instance.get('frame_start'),
                'frame_end': instance.get('frame_end'),
                'fps': instance.get('fps', 25)
            }
            gloss_to_instances[gloss_name].append(video_id)
        
        gloss_id += 1
    
    print(f"  Total video instances: {len(video_to_gloss)}")
    print(f"  Sample mapping: {list(video_to_gloss.items())[:3]}")
    
    # Find actual video files
    print("\n[STEP 3] Finding actual video files...")
    available_videos = set()
    if videos_src.exists():
        for video_file in videos_src.glob("*.mp4"):
            video_id = video_file.stem  # Get filename without extension
            available_videos.add(video_id)
    
    print(f"  Found {len(available_videos)} video files")
    
    # Filter videos that exist in both metadata and raw_videos
    mapped_videos = []
    for video_id in available_videos:
        if video_id in video_to_gloss:
            mapped_videos.append(video_id)
    
    print(f"  Videos with metadata: {len(mapped_videos)}")
    print(f"  Sample: {list(mapped_videos)[:5]}")
    
    # Shuffle and split (70/15/15)
    print("\n[STEP 4] Splitting into train/val/test (70/15/15)...")
    random.seed(42)
    random.shuffle(mapped_videos)
    
    train_split = int(len(mapped_videos) * 0.70)
    val_split = int(len(mapped_videos) * 0.85)
    
    train_videos = mapped_videos[:train_split]
    val_videos = mapped_videos[train_split:val_split]
    test_videos = mapped_videos[val_split:]
    
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val:   {len(val_videos)} videos")
    print(f"  Test:  {len(test_videos)} videos")
    
    # Copy and organize videos
    print("\n[STEP 5] Copying videos to train/val/test folders...")
    splits = {
        'train': (train_videos, train_dir),
        'val': (val_videos, val_dir),
        'test': (test_videos, test_dir)
    }
    
    metadata_splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for split_name, (video_ids, dest_dir) in splits.items():
        print(f"\n  [{split_name.upper()}]")
        for i, video_id in enumerate(video_ids):
            src_file = videos_src / f"{video_id}.mp4"
            dest_file = dest_dir / f"{video_id}.mp4"
            
            if src_file.exists():
                shutil.copy2(src_file, dest_file)
                gloss_info = video_to_gloss[video_id]
                metadata_splits[split_name].append({
                    'video_id': video_id,
                    'filename': f"{video_id}.mp4",
                    'gloss_id': gloss_info['gloss_id'],
                    'gloss': gloss_info['gloss_name'],
                    'bbox': gloss_info['bbox'],
                    'frame_start': gloss_info['frame_start'],
                    'frame_end': gloss_info['frame_end'],
                    'fps': gloss_info['fps']
                })
            
            if (i + 1) % 50 == 0:
                print(f"    Copied {i + 1}/{len(video_ids)} videos")
        
        print(f"    ✓ Completed {split_name.upper()}")
    
    # Create metadata.json
    print("\n[STEP 6] Creating metadata.json...")
    metadata = {
        'dataset': 'WLASL',
        'version': '0.3',
        'total_videos': len(mapped_videos),
        'total_glosses': len(gloss_vocab),
        'splits': {
            'train': {
                'count': len(train_videos),
                'videos': metadata_splits['train']
            },
            'val': {
                'count': len(val_videos),
                'videos': metadata_splits['val']
            },
            'test': {
                'count': len(test_videos),
                'videos': metadata_splits['test']
            }
        },
        'gloss_vocab': gloss_vocab
    }
    
    with open(data_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved metadata.json")
    
    # Create gloss_vocab.json (for easy reference)
    print("\n[STEP 7] Creating gloss_vocab.json...")
    gloss_vocab_inv = {v: k for k, v in gloss_vocab.items()}
    
    vocab_data = {
        'gloss_to_id': gloss_vocab_inv,
        'id_to_gloss': gloss_vocab,
        'total': len(gloss_vocab)
    }
    
    with open(data_dir / "gloss_vocab.json", 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"  ✓ Saved gloss_vocab.json with {len(gloss_vocab)} glosses")
    
    # Create dataset_config.json
    print("\n[STEP 8] Creating dataset_config.json...")
    dataset_config = {
        'name': 'WLASL',
        'version': '0.3',
        'total_videos': len(mapped_videos),
        'total_glosses': len(gloss_vocab),
        'splits': {
            'train': len(train_videos),
            'val': len(val_videos),
            'test': len(test_videos)
        },
        'vocab_size': len(gloss_vocab),
        'hand_input_dim': 126,
        'face_input_dim': 1404,
        'pose_input_dim': 99,
        'target_fps': 25
    }
    
    with open(data_dir / "dataset_config.json", 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"  ✓ Saved dataset_config.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total Videos:  {len(mapped_videos)}")
    print(f"  Total Glosses: {len(gloss_vocab)}")
    print(f"  Train Split:   {len(train_videos)} videos")
    print(f"  Val Split:     {len(val_videos)} videos")
    print(f"  Test Split:    {len(test_videos)} videos")
    print(f"\n📁 Files Created:")
    print(f"  data/train/          ({len(train_videos)} videos)")
    print(f"  data/val/            ({len(val_videos)} videos)")
    print(f"  data/test/           ({len(test_videos)} videos)")
    print(f"  data/metadata.json")
    print(f"  data/gloss_vocab.json")
    print(f"  data/dataset_config.json")
    print(f"\n🎯 Next Step: Update config.py with:")
    print(f"  vocab_size: {len(gloss_vocab)}")
    print(f"\n  Then run: python train.py")
    print("=" * 80)

if __name__ == "__main__":
    organize_wlasl_data()
