#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-prototype}"
MAX_WLASL="${MAX_WLASL:-48}"
MAX_HOW2SIGN="${MAX_HOW2SIGN:-48}"
MAX_FRAMES="${MAX_FRAMES:-32}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-8}"

echo "[1/5] Python and CUDA check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

echo "[2/5] Install dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[3/5] Prepare real WLASL + How2Sign data"
python prepare_real_datasets.py \
  --clear \
  --download \
  --max-wlasl "${MAX_WLASL}" \
  --max-how2sign "${MAX_HOW2SIGN}" \
  --max-frames "${MAX_FRAMES}"

echo "[4/5] Train"
if [[ "${MODE}" == "full" ]]; then
  python train_full_architecture.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-heads "${NUM_HEADS}"
else
  python train_prototype.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-heads "${NUM_HEADS}"
fi

echo "[5/5] Evaluate and save visuals/notebook"
if [[ "${MODE}" == "full" ]]; then
  python evaluate_full_architecture.py --split val --out-dir eval_outputs
else
  python evaluate_prototype.py --split val --out-dir eval_outputs
fi

echo "Done. Outputs:"
echo "  data/prototype_manifest.json"
echo "  checkpoints/"
echo "  eval_outputs/metrics.json"
echo "  eval_outputs/evaluation_report.ipynb"
