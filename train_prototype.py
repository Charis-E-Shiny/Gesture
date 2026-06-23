import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototype_dataset import CachedLandmarkCollator, CachedLandmarkDataset
from prototype_model import ASLLiveCaptionPrototype


def load_manifest(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct_gloss = 0
    correct_caption = 0
    total = 0

    for batch in tqdm(loader, desc="train" if train else "eval"):
        hand = batch["hand"].to(device)
        face = batch["face"].to(device)
        pose = batch["pose"].to(device)
        gloss_ids = batch["gloss_ids"].to(device)
        caption_ids = batch["caption_ids"].to(device)

        with torch.set_grad_enabled(train):
            output = model(hand, face, pose)
            gloss_loss = F.cross_entropy(output["gloss_logits"], gloss_ids)
            caption_loss = F.cross_entropy(output["caption_logits"], caption_ids)
            #loss = gloss_loss + caption_loss
            loss = gloss_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * hand.shape[0]
        correct_gloss += (output["gloss_logits"].argmax(dim=-1) == gloss_ids).sum().item()
        correct_caption += (output["caption_logits"].argmax(dim=-1) == caption_ids).sum().item()
        total += hand.shape[0]

    return {
        "loss": total_loss / max(total, 1),
        "gloss_acc": correct_gloss / max(total, 1),
        "caption_acc": correct_caption / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the small live caption prototype.")
    parser.add_argument("--manifest", default="data/prototype_manifest.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CachedLandmarkDataset(str(manifest_path), split="train")
    val_ds = CachedLandmarkDataset(str(manifest_path), split="val")
    collator = CachedLandmarkCollator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = ASLLiveCaptionPrototype(
        gloss_vocab_size=len(manifest["gloss_to_id"]),
        caption_vocab_size=len(manifest["caption_to_id"]),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
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
                "history": history,
            }, checkpoint_dir / "prototype_best.pt")

    with open(checkpoint_dir / "prototype_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint to {checkpoint_dir / 'prototype_best.pt'}")


if __name__ == "__main__":
    main()
