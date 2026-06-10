#!/usr/bin/env python3
"""
Setup script for ASL Recognition Pipeline

This script guides you through setting up the project.
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print formatted section"""
    print(f"\n▸ {text}")


def check_python_version():
    """Check Python version"""
    print_section("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python 3.8+ required (you have {version.major}.{version.minor})")
        return False


def check_gpu():
    """Check GPU availability"""
    print_section("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("⚠ GPU not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False


def create_directories():
    """Create necessary directories"""
    print_section("Creating project directories...")
    
    directories = [
        './data/train',
        './data/val',
        './data/test',
        './checkpoints',
        './logs',
        './cache',
        './uploads'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def install_dependencies():
    """Install dependencies"""
    print_section("Installing dependencies...")
    
    requirements_file = 'requirements.txt'
    
    if not Path(requirements_file).exists():
        print(f"✗ {requirements_file} not found")
        return False
    
    try:
        print("Installing PyTorch...")
        # Use CPU version if you don't have CUDA
        # For GPU, use: https://download.pytorch.org/whl/cu121
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ])
        print("✓ PyTorch installed")
        
        print("Installing other dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements_file
        ])
        print("✓ Dependencies installed")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False


def verify_installation():
    """Verify installation"""
    print_section("Verifying installation...")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'fastapi': 'FastAPI',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False
    
    return all_ok


def create_quickstart_guide():
    """Create quickstart guide"""
    print_section("Creating quickstart guide...")
    
    guide = '''# ASL Recognition Pipeline - Quick Start

## After Setup

### 1. Verify Installation
```bash
python config.py
```

### 2. Run Tests
```bash
python test_pipeline.py
```

### 3. Try Examples
```bash
python examples.py
```

### 4. Start Training (with data)
```bash
python train.py
```

### 5. Run Real-time Inference
```bash
python -c "
from inference import RealtimeInferencePipeline
pipeline = RealtimeInferencePipeline()
pipeline.run_webcam(camera_id=0)
"
```

### 6. Start API Server
```bash
python app.py
# Then visit http://localhost:8000/docs
```

## Project Files

- `models_encoders.py` - Hand, Face, Pose encoders
- `models_fusion.py` - Cross-modal fusion transformer
- `models_conformer.py` - Conformer streaming encoder
- `models_rnnt.py` - RNNT transducer
- `models_translator.py` - Seq2Seq translator
- `data_extractor.py` - MediaPipe & dataset utilities
- `pipeline.py` - Complete pipeline
- `train.py` - Training script
- `inference.py` - Real-time inference
- `app.py` - FastAPI application
- `config.py` - Configuration & utilities
- `test_pipeline.py` - Test suite
- `examples.py` - Usage examples

## Data Structure

Place your data in:
```
data/
├── train/
│   └── gloss_name/
│       ├── video_1.mp4
│       └── video_2.mp4
├── val/
└── test/
```

## Troubleshooting

### Import errors
Make sure you ran: `pip install -r requirements.txt`

### GPU not detected
Install CUDA: https://developer.nvidia.com/cuda-downloads

### MediaPipe issues
```bash
pip install --upgrade mediapipe
```

### Out of memory
Reduce batch_size in config.py

For more help, see README.md
'''
    
    with open('QUICKSTART.md', 'w') as f:
        f.write(guide)
    
    print("✓ Created QUICKSTART.md")


def main():
    """Main setup function"""
    print_header("ASL RECOGNITION PIPELINE - SETUP")
    
    print("\nThis setup script will:")
    print("  1. Check system requirements")
    print("  2. Create necessary directories")
    print("  3. Install dependencies")
    print("  4. Verify installation")
    print("  5. Create quick start guide")
    
    # Check Python version
    if not check_python_version():
        print_header("SETUP FAILED")
        print("Please install Python 3.8 or later")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    print_header("INSTALLING DEPENDENCIES")
    if not install_dependencies():
        print("\n⚠ Some dependencies failed to install")
        print("Try manually: pip install -r requirements.txt")
    
    # Verify installation
    print_header("VERIFYING INSTALLATION")
    if not verify_installation():
        print("\n⚠ Some packages not installed")
    
    # Create quickstart
    create_quickstart_guide()
    
    # Final message
    print_header("SETUP COMPLETE")
    print("""
✓ Project is ready!

Next steps:
  1. Read QUICKSTART.md for usage
  2. Run: python test_pipeline.py
  3. Try examples: python examples.py
  4. Prepare your data in ./data/
  5. Train: python train.py

For detailed info, see README.md
    """)


if __name__ == '__main__':
    main()
