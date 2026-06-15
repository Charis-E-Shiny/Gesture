# AUTOMATED WLASL Dataset Setup - Complete Guide

## 🚀 Quick Start (No Manual Work!)

### Option 1: Complete Automated Setup (Recommended)

```bash
# Run the automated setup - handles EVERYTHING
python setup_wlasl_dataset.py
```

This single command will:
- ✅ Clone WLASL repository
- ✅ Download videos from WLASL dataset
- ✅ Organize into train/val/test splits (70/15/15)
- ✅ Validate all videos
- ✅ Generate metadata JSON
- ✅ Create vocabulary files
- ✅ Generate training config
- ✅ Clean up temporary files

**Time**: 2-6 hours (depending on number of glosses and your internet)

---

### Option 2: Faster Setup (Limited Glosses)

Want to test quickly? Download only first 100 glosses:

```bash
python setup_wlasl_dataset.py --num-glosses 100
```

**Time**: 30 minutes - 1 hour

---

### Option 3: Custom Configuration

```bash
# Use only first 50 glosses, require 20+ videos each
python setup_wlasl_dataset.py --num-glosses 50 --min-videos 20

# Use custom output directory
python setup_wlasl_dataset.py --output ./my_dataset

# All options combined
python setup_wlasl_dataset.py --num-glosses 200 --min-videos 15 --output ./data_200glosses
```

---

## 📊 What Gets Created

After running the setup, you'll have:

```
data/
├── train/
│   ├── hello/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── ...
│   ├── thank_you/
│   │   ├── video_001.mp4
│   │   └── ...
│   └── (more glosses)
│
├── val/
│   ├── hello/
│   ├── thank_you/
│   └── (same structure)
│
├── test/
│   ├── hello/
│   ├── thank_you/
│   └── (same structure)
│
├── metadata.json         ← Video listing and paths
├── gloss_vocab.json      ← Vocabulary (gloss to ID mapping)
├── dataset_config.json   ← Dataset statistics
└── README.md             ← Dataset info
```

---

## ✅ What Happens Automatically

### Step 1: Clone WLASL Repository
The script clones the official WLASL GitHub repository which contains:
- Video URL listings
- Metadata
- Download scripts

### Step 2: Download Videos
Uses `yt-dlp` to download videos from YouTube links in WLASL:
- Automatic installation if needed
- Parallel downloads (fast!)
- Retry on failure
- Progress tracking

### Step 3: Organize into Splits
Automatically:
- Creates train (70%), val (15%), test (15%) splits
- Shuffles videos randomly
- Creates proper directory structure
- Maintains gloss integrity (same gloss across all splits)

### Step 4: Validate
Checks:
- All files are readable
- Videos can be opened
- Frame rates and dimensions
- No corrupted files

### Step 5: Generate Metadata
Creates:
- `metadata.json` - Complete video listing
- `gloss_vocab.json` - Vocabulary mapping
- `dataset_config.json` - Statistics and config

### Step 6: Clean Up
Removes:
- Temporary WLASL repository
- Temporary download folders
- Keeps only organized dataset

---

## 🎯 Training After Setup

Once setup is complete, training is automatic:

```bash
# Update vocab size in config.py (shown in setup output)
# Then start training
python train.py
```

The training script will:
- Auto-detect dataset in `./data`
- Load metadata from `metadata.json`
- Extract landmarks from videos
- Train the RNNT model
- Save checkpoints

---

## 📈 Example Output

```
======================================================================
  AUTOMATED WLASL DATASET SETUP
======================================================================

▸ STEP 1: Cloning WLASL Repository...
  ✓ Cloned successfully

▸ STEP 2: Loading WLASL Glosses...
  ✓ Found 2000 glosses

▸ STEP 3: Creating Directory Structure...
  ✓ Created ./data/train
  ✓ Created ./data/val
  ✓ Created ./data/test

▸ STEP 4: Downloading Videos...
  Running WLASL download script...
  ✓ Download completed

▸ STEP 5: Organizing Videos into Train/Val/Test Splits...
  • hello: 50 videos → (35/7/8)
  • thank_you: 48 videos → (34/7/7)
  • yes: 52 videos → (36/8/8)
  ...
  ✓ Organized 5000 videos from 100 glosses

▸ STEP 6: Validating Dataset...
  Dataset Statistics:
    TRAIN: Videos: 3500, Glosses: 100
    VAL: Videos: 750, Glosses: 100
    TEST: Videos: 750, Glosses: 100
  Total Videos: 5000
  Total Glosses: 100
  ✓ Validation complete

▸ STEP 7: Generating Metadata...
  ✓ Saved metadata to ./data/metadata.json
    Glosses: 100
    Total videos: 5000

▸ STEP 8: Creating Vocabulary Files...
  ✓ Created gloss vocabulary: 103 glosses
  ✓ Vocabulary file: ./data/gloss_vocab.json

▸ STEP 9: Creating Training Configuration...
  ✓ Config file: ./data/dataset_config.json

▸ STEP 10: Cleaning Up Temporary Files...
  ✓ Removed temporary WLASL repo
  ✓ Removed temporary downloads folder
  ✓ Cleanup complete

======================================================================
✓ SETUP COMPLETE!
======================================================================

Dataset Location: ./data

Dataset Statistics:
  - Total Glosses: 100
  - Total Videos: 5000
  - Train: 3500 videos
  - Val: 750 videos
  - Test: 750 videos

Next Steps:
  1. Update config.py with vocabulary size: 103
  2. Run training: python train.py
  3. Monitor: tensorboard --logdir ./logs
```

---

## 🔧 Troubleshooting

### Issue: "git not found"
```bash
# Install Git from: https://git-scm.com/download/win
# Or use: choco install git
```

### Issue: "yt-dlp failed"
The script auto-installs it, but if you want manual:
```bash
pip install yt-dlp
```

### Issue: Downloads are very slow
```bash
# Use fewer glosses for testing
python setup_wlasl_dataset.py --num-glosses 50
```

### Issue: Out of disk space
```bash
# The videos take ~100GB per 1000 glosses
# Use fewer glosses:
python setup_wlasl_dataset.py --num-glosses 100  # ~10GB
```

### Issue: Some videos fail to download
```bash
# This is normal - some YouTube videos disappear
# The script will skip them and continue
# You can manually add videos later
```

---

## 📊 Disk Space Requirements

| Glosses | Videos | Size | Time |
|---------|--------|------|------|
| 50 | 500 | ~5GB | 20 min |
| 100 | 1000 | ~10GB | 40 min |
| 500 | 5000 | ~50GB | 3-4 hours |
| 1000 | 10000 | ~100GB | 6-8 hours |
| 2000 | 20000 | ~200GB | 12+ hours |

---

## 🎬 What's in WLASL?

WLASL contains:
- **2000+ unique ASL signs (glosses)**
- **21000+ videos** from signers
- **Multiple signers per gloss** (variation)
- **High quality** (mostly HD)
- **Complete metadata** (standardized)
- **YouTube-hosted** (automatic download)

Perfect for training a comprehensive ASL system!

---

## ⚙️ Advanced: Custom Download

If you want to download specific glosses only:

```bash
python setup_wlasl_dataset.py --num-glosses 20
```

This will download the first 20 glosses from WLASL.

Or, if you have your own videos:

```bash
# Place them in this structure:
# data/train/hello/video1.mp4
# data/train/hello/video2.mp4
# data/val/hello/video1.mp4
# data/test/hello/video1.mp4

# Then just run training:
python train.py
```

---

## 🚀 After Setup: Training

```bash
# Make sure config.py has correct vocab size
# Edit config.py and set:
# MODEL_CONFIG['vocab_size'] = 103  (from setup output)

# Then start training:
python train.py

# Monitor progress:
tensorboard --logdir ./logs
```

Training will:
- Load metadata from `./data/metadata.json`
- Extract landmarks automatically
- Train RNNT for gloss recognition
- Save best model to `./checkpoints/best.pt`
- Complete in 1-2 days (depending on dataset size)

---

## 📝 Summary

**You need to do:**
1. Run: `python setup_wlasl_dataset.py`
2. Wait for completion (2-6 hours)
3. Run: `python train.py`
4. Wait for training (1-2 days)

**That's it!** Everything else is automatic:
- ✅ Cloning
- ✅ Downloading
- ✅ Organizing
- ✅ Validating
- ✅ Generating configs
- ✅ Extracting landmarks
- ✅ Training

No manual work, no code changes, fully automated! 🎉

---

## 📞 Need Help?

If something fails:

```bash
# Run with verbose output:
python setup_wlasl_dataset.py --verbose

# Check what's in the data directory:
dir /s data\

# Validate existing data:
python -c "from data_extractor import *; validate_dataset()"

# Check dataset statistics:
python -c "
import json
with open('data/metadata.json') as f:
    m = json.load(f)
    for s in ['train','val','test']:
        print(f'{s}: {m[\"splits\"][s][\"count\"]} videos')
"
```

---

**Ready?** Just run: `python setup_wlasl_dataset.py` 🚀
