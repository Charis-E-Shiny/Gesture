# ASL Recognition & Translation System

A complete implementation of the ASL (American Sign Language) recognition and real-time translation system with the architecture shown in the specification.

## Architecture Overview

```
Webcam Stream (30-60 FPS)
    ↓
MediaPipe Holistic (landmark extraction)
    ↓
┌─────────────────────────────────┐
│ Hand Encoder  │ Face Encoder  │ Pose Encoder
└─────────────────────────────────┘
    ↓
Cross-modal Fusion Transformer (concat + cross-attention, 6 heads)
    ↓
Conformer Streaming Encoder (Conv + self-attention + causal masking)
    ↓
RNNT Transducer (prediction network + joint network)
    ↓
ASL Gloss Recognition
    ↓
ASL→English Translator (Seq2Seq)
    ↓
Live Caption Renderer (<500ms latency)
```

## Components

### 1. **models_encoders.py** - Individual Modality Encoders
- `HandEncoder`: Processes 21 hand landmarks (×3 coordinates = 63D)
- `FaceEncoder`: Processes 468 face mesh points (×3 = 1404D)
- `PoseEncoder`: Processes 33 body landmarks (×3 = 99D)

### 2. **models_fusion.py** - Cross-modal Fusion
- `MultiHeadCrossAttention`: 6-head cross-attention mechanism
- `CrossModalFusionTransformer`: Fuses hand, face, and pose representations

### 3. **models_conformer.py** - Streaming Encoder
- `ConformerBlock`: Combines convolution, self-attention, and feedforward
- `ConformerStreamingEncoder`: Streaming-capable encoder with causal masking

### 4. **models_rnnt.py** - RNNT Transducer
- `RNNTTransducer`: Combines encoder output with prediction network
- `RNNTLoss`: CTC-style loss for ASL recognition

### 5. **models_translator.py** - ASL→English Translation
- `TransformerTranslator`: Seq2Seq transformer for gloss-to-English translation
- `TranslationLoss`: Cross-entropy loss with label smoothing

### 6. **data_extractor.py** - Data Processing
- `MediaPipeLandmarkExtractor`: Extracts landmarks from video frames
- `LandmarkDataset`: PyTorch dataset for training
- `LandmarkCollator`: Handles variable-length sequences

### 7. **pipeline.py** - Complete Pipeline
- `ASLRecognitionPipeline`: Combines all components end-to-end
- Supports both training and inference modes

### 8. **train.py** - Training Loop
- `Trainer`: Manages training, validation, checkpointing
- Distributed training support

### 9. **inference.py** - Real-time Inference
- `RealtimeInferencePipeline`: Real-time processing from webcam/video
- Frame buffering for streaming
- Latency tracking

### 10. **app.py** - FastAPI Web Server
- REST API endpoints for inference
- WebSocket for real-time streaming
- Video upload and processing

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone repository
cd Gesture

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import mediapipe; print('MediaPipe: OK')"
```

## Usage

### 1. Training

```bash
python train.py
```

Configuration can be modified in the script. The trainer will:
- Extract landmarks from videos using MediaPipe
- Train the RNNT transducer for gloss recognition
- Train the Seq2Seq translator for English translation
- Save checkpoints to `./checkpoints/best.pt`

### 2. Real-time Inference (Webcam)

```python
from inference import RealtimeInferencePipeline

pipeline = RealtimeInferencePipeline(model_path='./checkpoints/best.pt')
pipeline.run_webcam(camera_id=0)
```

Features:
- 30 FPS webcam capture
- Real-time landmark extraction
- Streaming inference with causal attention
- Live caption rendering
- Latency tracking (<500ms target)

### 3. Video Processing

```python
from inference import RealtimeInferencePipeline

pipeline = RealtimeInferencePipeline(model_path='./checkpoints/best.pt')
results = pipeline.process_video_file('input.mp4', output_path='output_with_captions.mp4')
```

### 4. FastAPI Server

```bash
# Start server
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /infer_frame` - Single frame inference
- `WebSocket /ws` - Real-time streaming
- `POST /upload_video` - Video upload and processing

### 5. Using via Python

```python
from pipeline import ASLRecognitionPipeline
import torch

# Create model
model = ASLRecognitionPipeline(vocab_size=1000, english_vocab_size=5000)

# Load checkpoint
checkpoint = torch.load('./checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
hand = torch.randn(1, 30, 126)  # (batch, seq_len, hand_dim)
face = torch.randn(1, 30, 1404)  # (batch, seq_len, face_dim)
pose = torch.randn(1, 30, 99)   # (batch, seq_len, pose_dim)

output = model(hand, face, pose)
gloss_ids = torch.argmax(output['gloss_logits'], dim=-1)
english_ids = model.translate_gloss(gloss_ids)
```

## Dataset Format

Expected directory structure:
```
data/
├── train/
│   └── gloss_name/
│       ├── video_01.mp4
│       └── video_02.mp4
├── val/
│   └── gloss_name/
│       └── video_01.mp4
└── test/
    └── gloss_name/
        └── video_01.mp4
```

Supported ASL datasets:
- WLASL (ASL Dataset with full gloss annotations)
- ASLSee
- Custom video datasets with gloss and English translations

## Model Architecture Details

### Encoders
- **Input**: Raw landmark coordinates (hand, face, pose)
- **Processing**: Linear embedding → Positional encoding → 2-layer Transformer
- **Output**: 512-dim sequence embeddings

### Fusion Transformer
- **Input**: 3 × 512-dim sequences (hand, face, pose)
- **Process**: Concat to 1536-dim → 6-head cross-attention
- **Output**: 512-dim fused representation

### Conformer Encoder
- **Components**: Conv1D + Depthwise conv + GLU + Self-attention
- **Features**: Causal masking for streaming, residual connections
- **Layers**: 4 conformer blocks

### RNNT Transducer
- **Prediction Network**: 2-layer LSTM on previous gloss tokens
- **Joint Network**: Concat encoder output + prediction → tanh → Linear
- **Loss**: CTC-style loss with blank token

### Seq2Seq Translator
- **Encoder**: Processes gloss tokens
- **Decoder**: Generates English text with causal masking
- **Attention**: Multi-head cross-attention between encoder and decoder

## Performance Targets

- **Latency**: <500ms for end-to-end inference
- **FPS**: 30-60 frames per second from webcam
- **Accuracy**: Dataset-dependent (typical WER: 15-30%)

## Configuration

Key hyperparameters in `train.py`:

```python
# Model
vocab_size = 1000                    # ASL gloss vocabulary
english_vocab_size = 5000            # English vocabulary
hidden_dim = 512                     # Model dimension

# Training
learning_rate = 1e-3
batch_size = 4
num_epochs = 100
patience = 10                        # Early stopping

# Data
max_buffer_size = 30                 # Frames for streaming
```

## Troubleshooting

### GPU Memory Issues
```python
# Reduce batch size in train.py
batch_size = 2

# Or use gradient accumulation
```

### Inference Latency
```python
# Reduce max_buffer_size for less latency
pipeline.max_buffer_size = 10  # Down from 30
```

### Model Not Found
```bash
# Ensure checkpoints directory exists
mkdir -p ./checkpoints

# Download pre-trained weights (if available)
# wget <model-url> -O ./checkpoints/best.pt
```

## Future Improvements

1. **Multi-hand gestures**: Enhanced hand-hand interactions
2. **Facial expressions**: Better integration of face features
3. **Context modeling**: Temporal attention across longer sequences
4. **Multi-lingual translation**: Support for other sign languages
5. **Mobile deployment**: ONNX export and mobile optimization
6. **User feedback loop**: Active learning from corrections

## References

- MediaPipe: https://github.com/google/mediapipe
- Conformer: https://arxiv.org/abs/2005.08100
- RNNT: https://arxiv.org/abs/1811.06621
- Transformer: https://arxiv.org/abs/1706.03762

## License

MIT License

## Citation

If you use this project, please cite:

```bibtex
@software{asl_recognition_2024,
  title={ASL Recognition and Translation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/asl-recognition}
}
```
