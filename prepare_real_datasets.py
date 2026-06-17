import argparse
import csv
import json
import os
import random
import shutil
import stat
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from data_extractor import MediaPipeLandmarkExtractor


WLASL_HANDLE = "risangbaskoro/wlasl-processed"
HOW2SIGN_HANDLE = "rabeyaakter23/how2sign-i3d-features-mediapipe-features"
LANDMARK_WIDTH = 126 + 1404 + 99


def download_kaggle_dataset(handle: str, output_dir: Path) -> Path:
    import kagglehub

    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(kagglehub.dataset_download(handle, output_dir=str(output_dir)))


def clear_directory(path: Path):
    def onerror(func, failed_path, _exc_info):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    if path.exists():
        shutil.rmtree(path, onerror=onerror)


def split_landmarks(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    if array.shape[1] < LANDMARK_WIDTH:
        padded = np.zeros((array.shape[0], LANDMARK_WIDTH), dtype=np.float32)
        padded[:, : array.shape[1]] = array
        array = padded
    elif array.shape[1] > LANDMARK_WIDTH:
        array = array[:, :LANDMARK_WIDTH]

    return array[:, :126], array[:, 126:1530], array[:, 1530:1629]


def save_feature(path: Path, hand: np.ndarray, face: np.ndarray, pose: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hand=hand, face=face, pose=pose)


def valid_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"} and path.stat().st_size > 1024


def load_wlasl_metadata(root: Path) -> Dict[str, Dict]:
    candidates = list(root.rglob("*.json"))
    by_video_id: Dict[str, Dict] = {}
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, list):
            for gloss_entry in data:
                gloss = gloss_entry.get("gloss")
                for instance in gloss_entry.get("instances", []):
                    video_id = str(instance.get("video_id") or instance.get("id") or "")
                    if video_id and gloss:
                        by_video_id[video_id] = {
                            "gloss": gloss,
                            "split": instance.get("split", "train"),
                        }
        elif isinstance(data, dict) and "splits" in data:
            for split, split_data in data["splits"].items():
                for video in split_data.get("videos", []):
                    video_id = Path(video["filename"]).stem
                    by_video_id[video_id] = {
                        "gloss": video["gloss"],
                        "split": split,
                    }
    return by_video_id


def prepare_wlasl(root: Path, out_dir: Path, max_samples: int,
                  max_frames: int, seed: int) -> List[Dict]:
    metadata = load_wlasl_metadata(root)
    videos = [path for path in root.rglob("*") if path.is_file() and valid_video(path)]
    random.Random(seed).shuffle(videos)

    extractor = MediaPipeLandmarkExtractor()
    samples = []
    for video_path in videos:
        if len(samples) >= max_samples:
            break
        item = metadata.get(video_path.stem)
        if not item:
            gloss = video_path.parent.name
            if gloss.lower() in {"train", "val", "test", "videos", "data"}:
                continue
            item = {"gloss": gloss, "split": "train"}

        try:
            hand, face, pose = extractor.process_video(str(video_path), max_frames=max_frames)
        except Exception as exc:
            print(f"Skipping WLASL video {video_path}: {exc}")
            continue
        if len(hand) == 0:
            continue

        feature_rel = Path("features") / "wlasl" / f"{video_path.stem}.npz"
        save_feature(out_dir / feature_rel, hand, face, pose)
        samples.append({
            "dataset": "wlasl",
            "split": normalize_split(item.get("split", "train")),
            "gloss": item["gloss"],
            "caption": gloss_to_caption(item["gloss"]),
            "feature_path": feature_rel.as_posix(),
            "source_path": str(video_path),
        })

    return samples


def normalize_split(split: str) -> str:
    split = (split or "train").lower()
    if split in {"val", "valid", "validation"}:
        return "val"
    if split == "test":
        return "test"
    return "train"


def gloss_to_caption(gloss: str) -> str:
    return gloss.replace("_", " ").replace("-", " ").lower().strip()


def read_transcripts(root: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for path in list(root.rglob("*.csv")) + list(root.rglob("*.tsv")):
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    keys = {k.lower(): k for k in row.keys() if k}
                    id_key = next((keys[k] for k in keys if k in {"id", "video_id", "name", "file", "filename"}), None)
                    text_key = next((keys[k] for k in keys if k in {"text", "caption", "transcript", "translation", "sentence"}), None)
                    if id_key and text_key and row.get(text_key):
                        transcripts[Path(row[id_key]).stem] = row[text_key].strip()
        except Exception:
            continue
    return transcripts


def load_feature_array(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            for key in ("landmarks", "mediapipe", "keypoints", "features", "arr_0"):
                if key in data:
                    return data[key]
    except Exception as exc:
        print(f"Skipping feature {path}: {exc}")
    return None


def prepare_how2sign(root: Path, out_dir: Path, max_samples: int,
                     seed: int) -> List[Dict]:
    transcripts = read_transcripts(root)
    feature_files = [
        path for path in root.rglob("*")
        if path.suffix.lower() in {".npy", ".npz"} and path.is_file()
    ]
    random.Random(seed).shuffle(feature_files)

    samples = []
    for feature_path in feature_files:
        if len(samples) >= max_samples:
            break
        caption = transcripts.get(feature_path.stem)
        if not caption:
            # The feature mirrors usually encode the sentence id in the file name.
            caption = feature_path.stem.replace("_", " ").replace("-", " ")

        array = load_feature_array(feature_path)
        if array is None:
            continue
        hand, face, pose = split_landmarks(array)
        if len(hand) == 0:
            continue

        feature_rel = Path("features") / "how2sign" / f"{feature_path.stem}.npz"
        save_feature(out_dir / feature_rel, hand, face, pose)
        samples.append({
            "dataset": "how2sign",
            "split": "train" if len(samples) % 5 else "val",
            "gloss": "continuous_sign",
            "caption": caption,
            "feature_path": feature_rel.as_posix(),
            "source_path": str(feature_path),
        })

    return samples


def assign_ids(samples: List[Dict]) -> Dict:
    glosses = ["<blank>", "<unk>"] + sorted({sample["gloss"] for sample in samples})
    captions = ["<blank>"] + sorted({sample["caption"] for sample in samples})
    gloss_to_id = {gloss: idx for idx, gloss in enumerate(glosses)}
    caption_to_id = {caption: idx for idx, caption in enumerate(captions)}

    for sample in samples:
        sample["gloss_id"] = gloss_to_id.get(sample["gloss"], 1)
        sample["caption_id"] = caption_to_id[sample["caption"]]

    return {
        "gloss_to_id": gloss_to_id,
        "caption_to_id": caption_to_id,
        "samples": samples,
    }


def ensure_splits(samples: List[Dict]):
    if not samples:
        return
    by_split = {split: [s for s in samples if s["split"] == split] for split in ("train", "val", "test")}
    if not by_split["val"] and len(samples) > 1:
        samples[-1]["split"] = "val"
    if not by_split["test"] and len(samples) > 2:
        samples[-2]["split"] = "test"


def main():
    parser = argparse.ArgumentParser(description="Prepare real WLASL + How2Sign prototype data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--wlasl-root", default=None)
    parser.add_argument("--how2sign-root", default=None)
    parser.add_argument("--wlasl-handle", default=WLASL_HANDLE)
    parser.add_argument("--how2sign-handle", default=HOW2SIGN_HANDLE)
    parser.add_argument("--max-wlasl", type=int, default=24)
    parser.add_argument("--max-how2sign", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    if args.clear and data_dir.exists():
        clear_directory(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    wlasl_root = Path(args.wlasl_root) if args.wlasl_root else None
    how2sign_root = Path(args.how2sign_root) if args.how2sign_root else None

    if args.download:
        if wlasl_root is None:
            wlasl_root = download_kaggle_dataset(args.wlasl_handle, raw_dir / "wlasl")
        if how2sign_root is None:
            how2sign_root = download_kaggle_dataset(args.how2sign_handle, raw_dir / "how2sign")

    samples: List[Dict] = []
    if wlasl_root and wlasl_root.exists():
        samples.extend(prepare_wlasl(wlasl_root, data_dir, args.max_wlasl, args.max_frames, args.seed))
    if how2sign_root and how2sign_root.exists():
        samples.extend(prepare_how2sign(how2sign_root, data_dir, args.max_how2sign, args.seed))

    if not samples:
        raise RuntimeError(
            "No real samples were prepared. Pass --download or provide --wlasl-root/--how2sign-root."
        )

    ensure_splits(samples)
    manifest = assign_ids(samples)
    manifest["sources"] = {
        "wlasl": str(wlasl_root) if wlasl_root else None,
        "how2sign": str(how2sign_root) if how2sign_root else None,
        "wlasl_handle": args.wlasl_handle,
        "how2sign_handle": args.how2sign_handle,
    }

    with open(data_dir / "prototype_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(data_dir / "gloss_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"gloss_to_id": manifest["gloss_to_id"]}, f, indent=2)
    with open(data_dir / "caption_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"caption_to_id": manifest["caption_to_id"]}, f, indent=2)
    with open(data_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": len(manifest["gloss_to_id"]),
            "caption_vocab_size": len(manifest["caption_to_id"]),
            "num_samples": len(samples),
        }, f, indent=2)

    print(f"Prepared {len(samples)} real samples in {data_dir}")
    print(f"Gloss classes: {len(manifest['gloss_to_id'])}")
    print(f"Caption classes: {len(manifest['caption_to_id'])}")


if __name__ == "__main__":
    main()
