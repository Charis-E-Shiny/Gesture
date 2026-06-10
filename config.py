import torch
import torch.nn as nn
from pathlib import Path

# Configuration file for the ASL Recognition Pipeline

# ============ Model Configuration ============

MODEL_CONFIG = {
    'vocab_size': 1000,                 # ASL gloss vocabulary size
    'english_vocab_size': 5000,         # English vocabulary size
    'hidden_dim': 512,                  # Model hidden dimension
    
    # Encoder configuration
    'hand_input_dim': 126,              # 21 landmarks × 3 × 2 hands
    'face_input_dim': 1404,             # 468 landmarks × 3
    'pose_input_dim': 99,               # 33 landmarks × 3
    
    # Transformer configuration
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.1,
    'feedforward_dim': 2048,
    
    # Conformer configuration
    'conformer_kernel_size': 31,
    'conformer_num_layers': 4,
    
    # RNNT configuration
    'rnnt_hidden_dim': 512,
    'rnnt_num_layers': 2,
}

# ============ Training Configuration ============

TRAINING_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    
    # Learning rate scheduler
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    
    # Early stopping
    'patience': 10,
    
    # Data
    'num_workers': 4,
    'pin_memory': True,
}

# ============ Inference Configuration ============

INFERENCE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_buffer_size': 30,              # Frames for streaming (1 sec at 30 FPS)
    'model_path': './checkpoints/best.pt',
    'confidence_threshold': 0.5,
    'target_fps': 30,
}

# ============ Data Configuration ============

DATA_CONFIG = {
    'train_dir': './data/train',
    'val_dir': './data/val',
    'test_dir': './data/test',
    'cache_dir': './cache',
    
    # Preprocessing
    'target_fps': 30,
    'frame_size': (480, 360),           # (height, width)
    'pad_value': 0.0,
}

# ============ API Configuration ============

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'workers': 1,
}

# ============ Paths ============

PATHS = {
    'root': Path('.'),
    'checkpoints': Path('./checkpoints'),
    'data': Path('./data'),
    'cache': Path('./cache'),
    'logs': Path('./logs'),
}

# Create directories
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ============ Vocabulary ============

# Placeholder vocabulary (would be loaded from actual data)
GLOSS_VOCAB = {
    0: '<blank>',
    1: '<unk>',
    2: '<pad>',
    **{i: f'gloss_{i}' for i in range(3, MODEL_CONFIG['vocab_size'])}
}

ENGLISH_VOCAB = {
    0: '<pad>',
    1: '<unk>',
    2: '<start>',
    3: '<end>',
    **{i: f'word_{i}' for i in range(4, MODEL_CONFIG['english_vocab_size'])}
}

# Reverse vocabularies
GLOSS_VOCAB_INV = {v: k for k, v in GLOSS_VOCAB.items()}
ENGLISH_VOCAB_INV = {v: k for k, v in ENGLISH_VOCAB.items()}

# ============ Helper Functions ============

def get_device():
    """Get the device to use for training/inference"""
    return torch.device(TRAINING_CONFIG['device'])


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_config():
    """Print all configurations"""
    print("=" * 80)
    print("ASL RECOGNITION PIPELINE CONFIGURATION")
    print("=" * 80)
    
    print("\n[MODEL]")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[TRAINING]")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[INFERENCE]")
    for key, value in INFERENCE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[DATA]")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n[PATHS]")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_config()
    
    # Test imports
    print("\nTesting imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import mediapipe
        print(f"✓ MediaPipe {mediapipe.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
        
        print("\n✓ All dependencies installed successfully!")
    
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
