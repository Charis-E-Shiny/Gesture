import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data_extractor import MediaPipeLandmarkExtractor
from prototype_model import ASLLiveCaptionPrototype


class LiveCaptioner:
    def __init__(self, checkpoint_path: str = "checkpoints/prototype_best.pt",
                 buffer_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        manifest = checkpoint["manifest"]
        self.id_to_gloss = {int(v): k for k, v in manifest["gloss_to_id"].items()}
        self.id_to_caption = {int(v): k for k, v in manifest["caption_to_id"].items()}
        self.model = ASLLiveCaptionPrototype(
            gloss_vocab_size=len(self.id_to_gloss),
            caption_vocab_size=len(self.id_to_caption),
            hidden_dim=checkpoint.get("hidden_dim", 96),
            num_heads=checkpoint.get("num_heads", 6),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.extractor = MediaPipeLandmarkExtractor()
        self.buffer_size = buffer_size
        self.hand, self.face, self.pose = [], [], []

    def process_frame(self, frame):
        start = time.time()
        landmarks = self.extractor.extract_landmarks(frame)
        self.hand.append(landmarks["hand"])
        self.face.append(landmarks["face"])
        self.pose.append(landmarks["pose"])
        self.hand = self.hand[-self.buffer_size:]
        self.face = self.face[-self.buffer_size:]
        self.pose = self.pose[-self.buffer_size:]

        result = {
            "gloss": "",
            "english": "",
            "latency_ms": (time.time() - start) * 1000,
            "frames": len(self.hand),
        }
        if len(self.hand) < min(4, self.buffer_size):
            return result

        hand = torch.from_numpy(np.asarray(self.hand, dtype=np.float32)).unsqueeze(0).to(self.device)
        face = torch.from_numpy(np.asarray(self.face, dtype=np.float32)).unsqueeze(0).to(self.device)
        pose = torch.from_numpy(np.asarray(self.pose, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gloss_id, caption_id = self.model.predict(hand, face, pose)
        result["gloss"] = self.id_to_gloss.get(int(gloss_id.item()), "<unk>")
        result["english"] = self.id_to_caption.get(int(caption_id.item()), "")
        result["latency_ms"] = (time.time() - start) * 1000
        return result


def draw(frame, result):
    cv2.putText(frame, f"Frames: {result['frames']}  Latency: {result['latency_ms']:.1f}ms",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
    if result["gloss"]:
        cv2.putText(frame, f"Gloss: {result['gloss']}",
                    (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
    if result["english"]:
        cv2.putText(frame, f"English: {result['english'][:80]}",
                    (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 255), 2)
    return frame


def run_webcam(captioner, camera_id: int):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        result = captioner.process_frame(frame)
        cv2.imshow("ASL Live Caption Prototype", draw(frame, result))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_video(captioner, video_path: str):
    cap = cv2.VideoCapture(video_path)
    results = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results.append(captioner.process_frame(frame))
    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Live English captioning from webcam or video.")
    parser.add_argument("--checkpoint", default="checkpoints/prototype_best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    captioner = LiveCaptioner(args.checkpoint)
    if args.video:
        results = run_video(captioner, args.video)
        for item in results[-10:]:
            print(item)
    else:
        run_webcam(captioner, args.camera)


if __name__ == "__main__":
    main()
