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
    """Dataset for ASL recognition"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: directory containing video files
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.extractor = MediaPipeLandmarkExtractor()
        
        # This is a placeholder - would be replaced with actual dataset loading
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset samples"""
        # Placeholder: load from data_dir
        video_files = list(self.data_dir.glob(f'{self.split}/**/*.mp4'))
        
        for video_file in video_files:
            # Extract gloss from filename or metadata
            gloss = video_file.stem  # Use filename as gloss
            self.samples.append((str(video_file), gloss))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, gloss = self.samples[idx]
        
        # Extract landmarks
        hand_lm, face_lm, pose_lm = self.extractor.process_video(video_path)
        
        # Concatenate landmarks
        landmarks = np.concatenate([hand_lm, face_lm, pose_lm], axis=1)
        
        return {
            'landmarks': torch.from_numpy(landmarks).float(),
            'gloss': gloss,
            'video_path': video_path
        }


class LandmarkCollator:
    """Custom collator for landmark data (handles variable length sequences)"""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        landmarks_list = [item['landmarks'] for item in batch]
        glosses = [item['gloss'] for item in batch]
        
        # Pad sequences to maximum length in batch
        max_len = max(l.shape[0] for l in landmarks_list)
        
        padded_landmarks = []
        lengths = []
        
        for landmarks in landmarks_list:
            padded = torch.full((max_len, landmarks.shape[1]),
                              self.pad_value, dtype=landmarks.dtype)
            padded[:landmarks.shape[0], :] = landmarks
            padded_landmarks.append(padded)
            lengths.append(landmarks.shape[0])
        
        return {
            'landmarks': torch.stack(padded_landmarks),
            'glosses': glosses,
            'lengths': torch.tensor(lengths, dtype=torch.long)
        }
