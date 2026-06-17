# Kubeflow / Terminal Run Guide

This repo is set up so GitHub contains code only. Datasets, checkpoints, and evaluation outputs are generated locally on the machine that runs the job.

## 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd Gesture
```

## 2. Configure Kaggle credentials

KaggleHub can read credentials from environment variables:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

In Kubeflow, put these in notebook-server environment variables or a mounted secret.

## 3. Run a small end-to-end prototype

```bash
bash scripts/kubeflow_full_run.sh
```

This will:

- install dependencies
- download small WLASL + How2Sign subsets from Kaggle
- cache real MediaPipe-style feature tensors under `data/`
- train with CUDA if `torch.cuda.is_available()`
- save checkpoints under `checkpoints/`
- save metrics, plots, and an `.ipynb` report under `eval_outputs/`

## 4. Run the full architecture path

```bash
MODE=full EPOCHS=10 BATCH_SIZE=4 MAX_WLASL=96 MAX_HOW2SIGN=96 bash scripts/kubeflow_full_run.sh
```

The full path trains:

- hand encoder
- face encoder
- pose encoder
- cross-modal fusion transformer
- conformer streaming encoder
- RNN-T gloss loss path
- transformer caption/translation loss path

## 5. Useful knobs

```bash
MAX_WLASL=48
MAX_HOW2SIGN=48
MAX_FRAMES=32
EPOCHS=5
BATCH_SIZE=4
HIDDEN_DIM=128
NUM_HEADS=8
MODE=prototype   # or full
```

Increase sample counts only after the small run finishes cleanly.

## 6. Live captioning

After `MODE=prototype` training:

```bash
python live_caption.py --checkpoint checkpoints/prototype_best.pt
```

For a saved video:

```bash
python live_caption.py --checkpoint checkpoints/prototype_best.pt --video path/to/video.mp4
```

The live runner uses the trained checkpoint and overlays predicted gloss plus English caption text.
