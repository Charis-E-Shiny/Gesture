import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline import ASLRecognitionPipeline
from prototype_dataset import CachedLandmarkCollator, CachedLandmarkDataset


def epoch_step(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total = 0
    gloss_correct = 0

    for batch in tqdm(loader, desc="train-full" if train else "eval-full"):
        hand = batch["hand"].to(device)
        face = batch["face"].to(device)
        pose = batch["pose"].to(device)
        gloss_targets = batch["gloss_ids"].unsqueeze(1).to(device)
        caption_targets = batch["caption_ids"].unsqueeze(1).to(device)

        with torch.set_grad_enabled(train):
            output = model(
                hand,
                face,
                pose,
                gloss_targets=gloss_targets,
                english_targets=caption_targets,
            )
            loss = model.get_total_loss(output)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        pred = output["gloss_logits"].mean(dim=1).argmax(dim=-1)
        gloss_correct += (pred == batch["gloss_ids"].to(device)).sum().item()
        total_loss += loss.item() * hand.shape[0]
        total += hand.shape[0]

    return {
        "loss": total_loss / max(total, 1),
        "gloss_acc": gloss_correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the full ASL architecture on cached real features.")
    parser.add_argument("--manifest", default="data/prototype_manifest.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--conformer-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_ds = CachedLandmarkDataset(str(manifest_path), split="train")
    val_ds = CachedLandmarkDataset(str(manifest_path), split="val")
    collator = CachedLandmarkCollator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = ASLRecognitionPipeline(
        vocab_size=len(manifest["gloss_to_id"]),
        english_vocab_size=max(len(manifest["caption_to_id"]), 4),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        fusion_layers=args.fusion_layers,
        conformer_layers=args.conformer_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_step(model, train_loader, optimizer, device, train=True)
        val_metrics = epoch_step(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "manifest": manifest,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "encoder_layers": args.encoder_layers,
                "fusion_layers": args.fusion_layers,
                "conformer_layers": args.conformer_layers,
                "history": history,
                "model_type": "full_architecture",
            }, checkpoint_dir / "full_architecture_best.pt")

    with open(checkpoint_dir / "full_architecture_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint to {checkpoint_dir / 'full_architecture_best.pt'}")


if __name__ == "__main__":
    main()
