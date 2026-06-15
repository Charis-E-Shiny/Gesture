from fileinput import filename
from importlib.metadata import metadata

from pydantic import json

import torch
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Dict, Optional
from pathlib import Path


class MediaPipeLandmarkExtractor:
    """Extract hand, face, and pose landmarks using MediaPipe Holistic"""
    
    def __init__(self):
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract landmarks from a single frame
        
        Args:
            frame: (H, W, 3) BGR image
        
        Returns:
            landmarks: dict with 'hand', 'face', 'pose' keys
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe
        results = self.holistic.process(rgb_frame)
        
        landmarks = {
            'hand': self._process_hand_landmarks(results),
            'face': self._process_face_landmarks(results),
            'pose': self._process_pose_landmarks(results)
        }
        
        return landmarks
    
    def _process_hand_landmarks(self, results) -> np.ndarray:
        """Extract hand landmarks (21 landmarks × 3 coords = 63)"""
        # Both hands combined
        hand_landmarks = []
        
        # Left hand
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            hand_landmarks.extend([0.0] * 63)
        
        # Right hand
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            hand_landmarks.extend([0.0] * 63)
        
        return np.array(hand_landmarks, dtype=np.float32)
    
    def _process_face_landmarks(self, results) -> np.ndarray:
        """Extract face mesh landmarks (468 landmarks × 3 coords = 1404)"""
        face_landmarks = []
        
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                face_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            face_landmarks.extend([0.0] * 1404)
        
        return np.array(face_landmarks, dtype=np.float32)
    
    def _process_pose_landmarks(self, results) -> np.ndarray:
        """Extract pose landmarks (33 landmarks × 3 coords = 99)"""
        pose_landmarks = []
        
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            pose_landmarks.extend([0.0] * 99)
        
        return np.array(pose_landmarks, dtype=np.float32)
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process video and extract all landmarks
        
        Returns:
            hand_landmarks: (seq_len, 126)
            face_landmarks: (seq_len, 1404)
            pose_landmarks: (seq_len, 99)
        """
        cap = cv2.VideoCapture(video_path)
        
        hand_seq = []
        face_seq = []
        pose_seq = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            landmarks = self.extract_landmarks(frame)
            hand_seq.append(landmarks['hand'])
            face_seq.append(landmarks['face'])
            pose_seq.append(landmarks['pose'])
            
            frame_count += 1
        
        cap.release()
        
        return (np.array(hand_seq, dtype=np.float32),
                np.array(face_seq, dtype=np.float32),
                np.array(pose_seq, dtype=np.float32))


class LandmarkDataset(torch.utils.data.Dataset):
    """Dataset for ASL recognition - supports WLASL and custom datasets"""
    
    def __init__(self, data_dir: str, split: str = 'train', metadata_file: str = None):
        """
        Args:
            data_dir: directory containing video files
            split: 'train', 'val', or 'test'
            metadata_file: optional path to metadata.json from WLASL setup
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.extractor = MediaPipeLandmarkExtractor()
        self.samples = []
        self.gloss_to_id = {}

        vocab_file = Path(data_dir).parent / "gloss_vocab.json"

        if vocab_file.exists():
            import json

            with open(vocab_file, "r") as f:
                vocab = json.load(f)

                self.gloss_to_id = vocab["gloss_to_id"]
        else:
            raise FileNotFoundError(f"Could not find {vocab_file}")

        self.id_to_gloss = {
            idx: gloss
            for gloss, idx in self.gloss_to_id.items()
        }

        print(f"Loaded {len(self.gloss_to_id)} gloss mappings")
        
        # Try to load from metadata if available
        if metadata_file and Path(metadata_file).exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file: str):
        import json

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if self.split in    metadata['splits']:

            for video_entry in metadata['splits'][self.split]['videos']:

                filename = video_entry['filename']
                gloss = video_entry['gloss']

                video_path = self.data_dir / filename

                if video_path.exists():

                    gloss_id = self.gloss_to_id.get(gloss, 1)

                    self.samples.append(
                        (
                            str(video_path),
                            gloss,
                            gloss_id
                        )
                    )

        print(f"Loaded {len(self.samples)} videos from {self.split} split")
    
    def _load_dataset(self):
        """Load dataset samples from directory structure"""
        # Supports structure: data/split/gloss_name/video.mp4
        video_files = list(self.data_dir.glob(f'{self.split}/**/*.mp4'))
        
        for video_file in video_files:
            # Extract gloss from parent folder
            gloss = video_file.parent.name
            gloss_id = self.gloss_to_id.get(gloss, 1)

            self.samples.append(
                (
                    str(video_file),
                    gloss,
                    gloss_id
                )
            )
        
        if len(self.samples) > 0:
            print(f"Loaded {len(self.samples)} videos from {self.data_dir}/{self.split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, gloss, gloss_id = self.samples[idx]
        
        try:
            # Extract landmarks
            hand_lm, face_lm, pose_lm = self.extractor.process_video(video_path)
            
            # Handle empty videos
            if len(hand_lm) == 0:
                # Return a dummy single frame if video is empty
                hand_lm = np.zeros((1, 126), dtype=np.float32)
                face_lm = np.zeros((1, 1404), dtype=np.float32)
                pose_lm = np.zeros((1, 99), dtype=np.float32)
            
            # Concatenate landmarks (seq_len, 126+1404+99)
            landmarks = np.concatenate([hand_lm, face_lm, pose_lm], axis=1)
            
            # gloss_id = self.gloss_to_id.get(gloss, self.gloss_to_id.get('<unk>', 1))

            return {
                'landmarks': torch.from_numpy(landmarks).float(),
                'gloss': gloss,
                'gloss_id': gloss_id,
                'video_path': video_path
            }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return zeros if there's an error
            landmarks = np.zeros((1, 1629), dtype=np.float32)
            gloss_id = self.gloss_to_id.get(gloss,
                                self.gloss_to_id.get('<unk>', 1))

            return {
                'landmarks': torch.from_numpy(landmarks).float(),
                'gloss': gloss,
                'gloss_id': gloss_id,
                'video_path': video_path
            }


class LandmarkCollator:
    def __init__(self, pad_value=0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        landmarks_list = [item['landmarks'] for item in batch]

        glosses = [item['gloss'] for item in batch]

        gloss_ids = torch.tensor(
            [item['gloss_id'] for item in batch],
            dtype=torch.long
        )

        max_len = max(l.shape[0] for l in landmarks_list)

        padded_landmarks = []
        lengths = []

        for landmarks in landmarks_list:
            padded = torch.full(
                (max_len, landmarks.shape[1]),
                self.pad_value,
                dtype=landmarks.dtype
            )

            padded[:landmarks.shape[0]] = landmarks

            padded_landmarks.append(padded)
            lengths.append(landmarks.shape[0])

        return {
            'landmarks': torch.stack(padded_landmarks),
            'glosses': glosses,
            'gloss_ids': gloss_ids,
            'lengths': torch.tensor(lengths, dtype=torch.long)
        }
