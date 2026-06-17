import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline import ASLRecognitionPipeline
from prototype_dataset import CachedLandmarkCollator, CachedLandmarkDataset


def save_bar(metrics, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["loss", "gloss_acc"], [metrics["loss"], metrics["gloss_acc"]], color=["#5568c3", "#13936b"])
    ax.set_title("Full Architecture Validation Metrics")
    ax.set_ylim(0, max(1.0, metrics["loss"]))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_notebook(metrics, plot_path, path):
    try:
        import nbformat as nbf
    except ImportError:
        return

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Full ASL Architecture Evaluation"),
        nbf.v4.new_code_cell(f"metrics = {json.dumps(metrics, indent=2)}\nmetrics"),
        nbf.v4.new_markdown_cell(f"![metrics]({plot_path.name})"),
    ]
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate full ASL architecture checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/full_architecture_best.pt")
    parser.add_argument("--manifest", default="data/prototype_manifest.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default="eval_outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    manifest = checkpoint["manifest"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASLRecognitionPipeline(
        vocab_size=len(manifest["gloss_to_id"]),
        english_vocab_size=max(len(manifest["caption_to_id"]), 4),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_heads=checkpoint.get("num_heads", 8),
        encoder_layers=checkpoint.get("encoder_layers", 1),
        fusion_layers=checkpoint.get("fusion_layers", 1),
        conformer_layers=checkpoint.get("conformer_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CachedLandmarkDataset(args.manifest, split=args.split)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=CachedLandmarkCollator())

    total_loss = 0.0
    total = 0
    correct = 0
    predictions = []
    with torch.no_grad():
        for batch in loader:
            hand = batch["hand"].to(device)
            face = batch["face"].to(device)
            pose = batch["pose"].to(device)
            gloss_targets = batch["gloss_ids"].unsqueeze(1).to(device)
            caption_targets = batch["caption_ids"].unsqueeze(1).to(device)
            output = model(hand, face, pose, gloss_targets=gloss_targets, english_targets=caption_targets)
            loss = model.get_total_loss(output)
            pred = output["gloss_logits"].mean(dim=1).argmax(dim=-1)
            correct += (pred == batch["gloss_ids"].to(device)).sum().item()
            total_loss += loss.item() * hand.shape[0]
            total += hand.shape[0]
            predictions.extend(pred.cpu().tolist())

    metrics = {
        "split": args.split,
        "num_samples": total,
        "loss": total_loss / max(total, 1),
        "gloss_acc": correct / max(total, 1),
        "predicted_gloss_ids": predictions,
    }
    with open(out_dir / "full_architecture_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_path = out_dir / "full_architecture_metrics.png"
    save_bar(metrics, plot_path)
    write_notebook(metrics, plot_path, out_dir / "full_architecture_report.ipynb")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
