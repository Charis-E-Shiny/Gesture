# ASL Live Captioning Prototype

Terminal-first ASL recognition and English captioning pipeline for WLASL + How2Sign experiments.

The repo keeps code in Git and generates datasets, checkpoints, and evaluation outputs at runtime. It is designed to run from a Kubeflow notebook-server terminal and uses CUDA automatically when PyTorch can see a GPU.

## Architecture

The full training path follows:

1. WLASL / How2Sign real data
2. MediaPipe-style hand, face, and pose features
3. Hand encoder, face encoder, pose encoder
4. Cross-modal fusion transformer
5. Conformer streaming encoder
6. RNN-T gloss recognition path
7. Transformer caption/translation path
8. Evaluation plots and notebook reports

For quick iteration, `MODE=prototype` trains a smaller live-caption head on the same encoded representation.

## Kubeflow Quick Run

Set Kaggle credentials in the notebook server environment or terminal:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Small end-to-end run:

```bash
bash scripts/kubeflow_full_run.sh
```

Full architecture run:

```bash
MODE=full EPOCHS=10 BATCH_SIZE=4 MAX_WLASL=96 MAX_HOW2SIGN=96 bash scripts/kubeflow_full_run.sh
```

## Useful Environment Variables

```bash
MODE=prototype   # prototype or full
MAX_WLASL=48
MAX_HOW2SIGN=48
MAX_FRAMES=32
EPOCHS=5
BATCH_SIZE=4
HIDDEN_DIM=128
NUM_HEADS=8
```

## Outputs

Generated files are intentionally ignored by Git:

- `data/`
- `checkpoints/`
- `eval_outputs/`

Evaluation scripts save metrics JSON, PNG plots, and `.ipynb` reports.

## Live Captioning

After prototype training:

```bash
python live_caption.py --checkpoint checkpoints/prototype_best.pt
```

For a saved video:

```bash
python live_caption.py --checkpoint checkpoints/prototype_best.pt --video path/to/video.mp4
```

See `KUBEFLOW_RUN.md` for a fuller terminal guide.
