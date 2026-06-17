#!/usr/bin/env bash
set -euo pipefail

MAX_WLASL="${MAX_WLASL:-48}"
MAX_HOW2SIGN="${MAX_HOW2SIGN:-48}"
MAX_FRAMES="${MAX_FRAMES:-32}"

python -m pip install -r requirements.txt
python prepare_real_datasets.py \
  --clear \
  --download \
  --max-wlasl "${MAX_WLASL}" \
  --max-how2sign "${MAX_HOW2SIGN}" \
  --max-frames "${MAX_FRAMES}"
