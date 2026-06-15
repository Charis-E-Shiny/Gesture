#!/usr/bin/env python3
"""
Fully Automated WLASL Dataset Setup
- Downloads WLASL repository
- Organizes videos into train/val/test splits
- Validates dataset
- Generates metadata JSON
- Ready to train!

No manual work required!
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
import random
import cv2


class WLASLSetup:
    """Automated WLASL dataset setup"""
    
    def __init__(self, output_dir="./data", num_glosses=None, min_videos_per_gloss=10):
        """
        Initialize setup
        
        Args:
            output_dir: Where to save organized dataset
            num_glosses: How many glosses to download (None = all)
            min_videos_per_gloss: Minimum videos per gloss to include
        """
        self.output_dir = Path(output_dir)
        self.num_glosses = num_glosses
        self.min_videos_per_gloss = min_videos_per_gloss
        self.wlasl_repo = Path("./WLASL_temp")
        
        # Split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def print_step(self, text):
        """Print step"""
        print(f"\n▸ {text}")
    
    def step1_clone_wlasl(self):
        """Step 1: Clone WLASL repository"""
        self.print_step("STEP 1: Cloning WLASL Repository...")
        
        if self.wlasl_repo.exists():
            print(f"  ✓ Already cloned at {self.wlasl_repo}")
            return
        
        print(f"  Cloning from GitHub...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/dxli94/WLASL.git", str(self.wlasl_repo)],
                check=True,
                capture_output=True
            )
            print(f"  ✓ Cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Clone failed: {e}")
            print(f"  Manual fallback: git clone https://github.com/dxli94/WLASL.git")
            sys.exit(1)
    
    def step2_get_glosses(self):
        """Step 2: Get list of available glosses"""
        self.print_step("STEP 2: Loading WLASL Glosses...")
        
        # WLASL has a vocab file with all glosses
        vocab_file = self.wlasl_repo / "nslt_2000.json"
        
        if not vocab_file.exists():
            print(f"  ✗ Vocab file not found at {vocab_file}")
            print(f"  Checking for alternative locations...")
            
            # Try common locations
            alternatives = [
                self.wlasl_repo / "start_kit" / "WLASL_v0.3.json",
                self.wlasl_repo / "WLASL_v0.3.json",
                self.wlasl_repo / "WLASL_v0.2.json",
                self.wlasl_repo / "data" / "nslt_2000.json"
            ]
            
            for alt in alternatives:
                if alt.exists():
                    vocab_file = alt
                    break
        
        if not vocab_file.exists():
            print(f"  ✗ Cannot find WLASL vocab file")
            print(f"  Please ensure WLASL is properly cloned")
            sys.exit(1)
        
        print(f"  Loading glosses from {vocab_file.name}...")
        
        with open(vocab_file, 'r') as f:
            self.glosses_data = json.load(f)
        
        total_glosses = len(self.glosses_data)
        print(f"  ✓ Found {total_glosses} glosses")
        
        # Limit glosses if specified
        if self.num_glosses:
            self.glosses_data = self.glosses_data[:self.num_glosses]
            print(f"  ✓ Using {len(self.glosses_data)} glosses (as specified)")
        
        return self.glosses_data
    
    def step3_create_directories(self):
        """Step 3: Create directory structure"""
        self.print_step("STEP 3: Creating Directory Structure...")
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {split_dir}")
    
    def step4_download_videos(self):
        """Step 4: Download videos using WLASL's download script"""
        self.print_step("STEP 4: Downloading Videos (this may take a while)...")
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(["yt-dlp", "--version"], 
                          capture_output=True, check=True)
            print(f"  ✓ yt-dlp is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  Installing yt-dlp...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"],
                             check=True)
                print(f"  ✓ yt-dlp installed")
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to install yt-dlp")
                sys.exit(1)
        
        # Use WLASL's download script if available
        download_script = self.wlasl_repo / "start_kit" / "video_downloader.py"
        
        if download_script.exists():
            print(f"  Running WLASL download script...")
            print(f"  (This will download {len(self.glosses_data)} glosses)")
            print(f"  NOTE: This takes several hours for large datasets!")
            
            try:
                # Create a temp folder for downloads
                temp_download = Path("./wlasl_downloads")
                temp_download.mkdir(exist_ok=True)
                
                # Run download script
                subprocess.run(
                    [sys.executable, str(download_script), 
                     "--output", str(temp_download),
                     "--glosses", ",".join([g['gloss'] for g in self.glosses_data[:50]]),  # Start with 50
                     "--format", "mp4"],
                    cwd=self.wlasl_repo,
                    capture_output=False
                )
                
                print(f"  ✓ Download completed")
                return temp_download
            
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Download script failed: {e}")
                print(f"  This might be due to YouTube availability")
                return None
        else:
            print(f"  ✗ Download script not found")
            print(f"  Please ensure WLASL repo structure is correct")
            return None
    
    def step5_organize_videos(self, download_dir):
        """Step 5: Organize downloaded videos into train/val/test"""
        self.print_step("STEP 5: Organizing Videos into Train/Val/Test Splits...")
        
        if not download_dir:
            print(f"  ✗ No download directory provided")
            return 0
        
        download_dir = Path(download_dir)
        total_videos = 0
        total_glosses = 0
        
        # Walk through downloaded videos
        for gloss_folder in os.listdir(download_dir):
            gloss_path = download_dir / gloss_folder
            
            if not gloss_path.is_dir():
                continue
            
            # Get all videos for this gloss
            videos = [f for f in os.listdir(gloss_path) 
                     if f.endswith(('.mp4', '.mkv', '.webm'))]
            
            if len(videos) < self.min_videos_per_gloss:
                print(f"  ⊘ {gloss_folder}: {len(videos)} videos (skipped, need {self.min_videos_per_gloss})")
                continue
            
            print(f"  • {gloss_folder}: {len(videos)} videos", end="")
            
            # Shuffle and split
            random.shuffle(videos)
            
            train_count = int(len(videos) * self.train_ratio)
            val_count = int(len(videos) * self.val_ratio)
            
            train_videos = videos[:train_count]
            val_videos = videos[train_count:train_count + val_count]
            test_videos = videos[train_count + val_count:]
            
            # Create gloss directories in each split
            for split in ['train', 'val', 'test']:
                split_dir = self.output_dir / split / gloss_folder
                split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            splits_count = {'train': 0, 'val': 0, 'test': 0}
            
            for video in train_videos:
                src = gloss_path / video
                dst = self.output_dir / 'train' / gloss_folder / video
                try:
                    shutil.copy2(src, dst)
                    splits_count['train'] += 1
                except Exception as e:
                    print(f"\n    Error copying {video}: {e}")
            
            for video in val_videos:
                src = gloss_path / video
                dst = self.output_dir / 'val' / gloss_folder / video
                try:
                    shutil.copy2(src, dst)
                    splits_count['val'] += 1
                except Exception as e:
                    print(f"\n    Error copying {video}: {e}")
            
            for video in test_videos:
                src = gloss_path / video
                dst = self.output_dir / 'test' / gloss_folder / video
                try:
                    shutil.copy2(src, dst)
                    splits_count['test'] += 1
                except Exception as e:
                    print(f"\n    Error copying {video}: {e}")
            
            total_videos += sum(splits_count.values())
            total_glosses += 1
            
            print(f" → ({splits_count['train']}/{splits_count['val']}/{splits_count['test']})")
        
        print(f"\n  ✓ Organized {total_videos} videos from {total_glosses} glosses")
        return total_videos
    
    def step6_validate_dataset(self):
        """Step 6: Validate organized dataset"""
        self.print_step("STEP 6: Validating Dataset...")
        
        stats = {
            'train': {'count': 0, 'glosses': set(), 'errors': []},
            'val': {'count': 0, 'glosses': set(), 'errors': []},
            'test': {'count': 0, 'glosses': set(), 'errors': []}
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            
            if not split_dir.exists():
                print(f"  ✗ {split} directory not found")
                continue
            
            for gloss_folder in os.listdir(split_dir):
                gloss_path = split_dir / gloss_folder
                
                if not gloss_path.is_dir():
                    continue
                
                videos = [f for f in os.listdir(gloss_path) 
                         if f.endswith(('.mp4', '.mkv', '.webm'))]
                
                stats[split]['count'] += len(videos)
                stats[split]['glosses'].add(gloss_folder)
                
                # Validate first video
                if videos:
                    first_video = gloss_path / videos[0]
                    try:
                        cap = cv2.VideoCapture(str(first_video))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                        else:
                            stats[split]['errors'].append(f"{gloss_folder}: Cannot open video")
                    except Exception as e:
                        stats[split]['errors'].append(f"{gloss_folder}: {str(e)}")
        
        # Print validation results
        print(f"\n  Dataset Statistics:")
        for split in ['train', 'val', 'test']:
            print(f"    {split.upper()}:")
            print(f"      Videos: {stats[split]['count']}")
            print(f"      Glosses: {len(stats[split]['glosses'])}")
            if stats[split]['errors']:
                print(f"      Errors: {len(stats[split]['errors'])}")
        
        total_videos = sum(s['count'] for s in stats.values())
        all_glosses = stats['train']['glosses'] | stats['val']['glosses'] | stats['test']['glosses']
        
        print(f"\n  Total Videos: {total_videos}")
        print(f"  Total Glosses: {len(all_glosses)}")
        
        if total_videos == 0:
            print(f"\n  ⚠ No videos found!")
            print(f"  Make sure download completed successfully")
            return False
        
        print(f"\n  ✓ Validation complete")
        return True
    
    def step7_generate_metadata(self):
        """Step 7: Generate metadata JSON"""
        self.print_step("STEP 7: Generating Metadata...")
        
        metadata = {
            'dataset': 'WLASL',
            'splits': {},
            'glosses': []
        }
        
        all_glosses = set()
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_videos = []
            
            if split_dir.exists():
                for gloss_folder in os.listdir(split_dir):
                    gloss_path = split_dir / gloss_folder
                    
                    if not gloss_path.is_dir():
                        continue
                    
                    all_glosses.add(gloss_folder)
                    
                    for video_file in os.listdir(gloss_path):
                        if video_file.endswith(('.mp4', '.mkv', '.webm')):
                            video_entry = {
                                'video_id': f"{gloss_folder}_{video_file}",
                                'gloss': gloss_folder,
                                'path': str(gloss_path / video_file).replace('\\', '/'),
                                'split': split
                            }
                            split_videos.append(video_entry)
            
            metadata['splits'][split] = {
                'count': len(split_videos),
                'videos': split_videos
            }
        
        metadata['glosses'] = sorted(list(all_glosses))
        
        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata to {metadata_file}")
        print(f"    Glosses: {len(metadata['glosses'])}")
        print(f"    Total videos: {sum(s['count'] for s in metadata['splits'].values())}")
        
        return metadata
    
    def step8_create_vocab(self, metadata):
        """Step 8: Create vocabulary files"""
        self.print_step("STEP 8: Creating Vocabulary Files...")
        
        # Create gloss vocabulary
        gloss_vocab = {0: '<blank>', 1: '<unk>', 2: '<pad>'}
        for idx, gloss in enumerate(sorted(metadata['glosses']), start=3):
            gloss_vocab[idx] = gloss
        
        # Save gloss vocab
        gloss_vocab_file = self.output_dir / 'gloss_vocab.json'
        with open(gloss_vocab_file, 'w') as f:
            json.dump(gloss_vocab, f, indent=2)
        
        print(f"  ✓ Created gloss vocabulary: {len(gloss_vocab)} glosses")
        
        # Create reverse vocab for config
        gloss_vocab_inv = {v: k for k, v in gloss_vocab.items()}
        config_update = {
            'vocab_size': len(gloss_vocab),
            'gloss_vocab': gloss_vocab,
            'gloss_vocab_inv': gloss_vocab_inv
        }
        
        print(f"  ✓ Vocabulary file: {gloss_vocab_file}")
        
        return gloss_vocab
    
    def step9_create_config(self, metadata, gloss_vocab):
        """Step 9: Create training config"""
        self.print_step("STEP 9: Creating Training Configuration...")
        
        config = {
            'dataset': {
                'name': 'WLASL',
                'data_dir': str(self.output_dir).replace('\\', '/'),
                'vocab_size': len(gloss_vocab),
                'num_glosses': len(metadata['glosses']),
                'total_videos': sum(s['count'] for s in metadata['splits'].values())
            },
            'splits': {
                'train': metadata['splits']['train']['count'],
                'val': metadata['splits']['val']['count'],
                'test': metadata['splits']['test']['count']
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 100,
                'learning_rate': 1e-3,
                'early_stopping_patience': 10
            }
        }
        
        config_file = self.output_dir / 'dataset_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Config file: {config_file}")
        
        return config
    
    def step10_cleanup(self):
        """Step 10: Cleanup temporary files"""
        self.print_step("STEP 10: Cleaning Up Temporary Files...")
        
        # Remove WLASL repo
        if self.wlasl_repo.exists():
            shutil.rmtree(self.wlasl_repo)
            print(f"  ✓ Removed temporary WLASL repo")
        
        # Remove download folder
        download_dir = Path("./wlasl_downloads")
        if download_dir.exists():
            shutil.rmtree(download_dir)
            print(f"  ✓ Removed temporary downloads folder")
        
        print(f"  ✓ Cleanup complete")
    
    def run(self):
        """Run complete setup"""
        self.print_header("AUTOMATED WLASL DATASET SETUP")
        
        try:
            # Step 1: Clone WLASL
            self.step1_clone_wlasl()
            
            # Step 2: Get glosses
            self.step2_get_glosses()
            
            # Step 3: Create directories
            self.step3_create_directories()
            
            # Step 4: Download videos
            download_dir = self.step4_download_videos()
            
            if not download_dir:
                print(f"\n⚠ Video download failed or skipped")
                print(f"Please download WLASL videos manually:")
                print(f"  1. Visit: https://github.com/dxli94/WLASL")
                print(f"  2. Follow their download instructions")
                print(f"  3. Place in: {self.output_dir}")
                return False
            
            # Step 5: Organize videos
            self.step5_organize_videos(download_dir)
            
            # Step 6: Validate
            if not self.step6_validate_dataset():
                return False
            
            # Step 7: Generate metadata
            metadata = self.step7_generate_metadata()
            
            # Step 8: Create vocabulary
            gloss_vocab = self.step8_create_vocab(metadata)
            
            # Step 9: Create config
            config = self.step9_create_config(metadata, gloss_vocab)
            
            # Step 10: Cleanup
            self.step10_cleanup()
            
            # Final summary
            self.print_header("✓ SETUP COMPLETE!")
            
            print(f"""
Dataset Location: {self.output_dir}

Dataset Statistics:
  - Total Glosses: {len(metadata['glosses'])}
  - Total Videos: {sum(s['count'] for s in metadata['splits'].values())}
  - Train: {metadata['splits']['train']['count']} videos
  - Val: {metadata['splits']['val']['count']} videos
  - Test: {metadata['splits']['test']['count']} videos

Next Steps:
  1. Update config.py with vocabulary size: {len(gloss_vocab)}
  2. Run training: python train.py
  3. Monitor: tensorboard --logdir ./logs

Files Created:
  ✓ {self.output_dir}/metadata.json
  ✓ {self.output_dir}/gloss_vocab.json
  ✓ {self.output_dir}/dataset_config.json
  ✓ {self.output_dir}/train/
  ✓ {self.output_dir}/val/
  ✓ {self.output_dir}/test/
            """)
            
            return True
        
        except Exception as e:
            print(f"\n✗ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated WLASL Dataset Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup all available glosses
  python setup_wlasl_dataset.py
  
  # Setup only first 100 glosses (faster)
  python setup_wlasl_dataset.py --num-glosses 100
  
  # Require at least 20 videos per gloss
  python setup_wlasl_dataset.py --min-videos 20
  
  # Custom output directory
  python setup_wlasl_dataset.py --output ./custom_data
        """
    )
    
    parser.add_argument('--output', '-o', default='./data',
                       help='Output directory for dataset (default: ./data)')
    parser.add_argument('--num-glosses', '-n', type=int, default=None,
                       help='Number of glosses to download (default: all)')
    parser.add_argument('--min-videos', '-m', type=int, default=10,
                       help='Minimum videos per gloss (default: 10)')
    
    args = parser.parse_args()
    
    # Run setup
    setup = WLASLSetup(
        output_dir=args.output,
        num_glosses=args.num_glosses,
        min_videos_per_gloss=args.min_videos
    )
    
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
