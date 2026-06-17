$ErrorActionPreference = "Stop"

$Mode = if ($env:MODE) { $env:MODE } else { "prototype" }
$MaxWlasl = if ($env:MAX_WLASL) { $env:MAX_WLASL } else { "48" }
$MaxHow2Sign = if ($env:MAX_HOW2SIGN) { $env:MAX_HOW2SIGN } else { "48" }
$MaxFrames = if ($env:MAX_FRAMES) { $env:MAX_FRAMES } else { "32" }
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "5" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "4" }
$HiddenDim = if ($env:HIDDEN_DIM) { $env:HIDDEN_DIM } else { "128" }
$NumHeads = if ($env:NUM_HEADS) { $env:NUM_HEADS } else { "8" }

Write-Host "[1/5] Python and CUDA check"
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

Write-Host "[2/5] Install dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "[3/5] Prepare real WLASL + How2Sign data"
python prepare_real_datasets.py --clear --download --max-wlasl $MaxWlasl --max-how2sign $MaxHow2Sign --max-frames $MaxFrames

Write-Host "[4/5] Train"
if ($Mode -eq "full") {
    python train_full_architecture.py --epochs $Epochs --batch-size $BatchSize --hidden-dim $HiddenDim --num-heads $NumHeads
} else {
    python train_prototype.py --epochs $Epochs --batch-size $BatchSize --hidden-dim $HiddenDim --num-heads $NumHeads
}

Write-Host "[5/5] Evaluate and save visuals/notebook"
if ($Mode -eq "full") {
    python evaluate_full_architecture.py --split val --out-dir eval_outputs
} else {
    python evaluate_prototype.py --split val --out-dir eval_outputs
}

Write-Host "Done. Outputs are in checkpoints/ and eval_outputs/"
