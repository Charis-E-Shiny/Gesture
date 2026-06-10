import torch
import cv2
import numpy as np
from typing import Dict, Tuple
import time

from pipeline import ASLRecognitionPipeline
from data_extractor import MediaPipeLandmarkExtractor


class RealtimeInferencePipeline:
    """Real-time inference pipeline for ASL recognition"""
    
    def __init__(self, model_path: str = './checkpoints/best.pt',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Load model
        self.model = ASLRecognitionPipeline(vocab_size=1000, 
                                           english_vocab_size=5000)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # MediaPipe extractor
        self.landmark_extractor = MediaPipeLandmarkExtractor()
        
        # Buffer for streaming
        self.landmark_buffer = {
            'hand': [],
            'face': [],
            'pose': []
        }
        self.max_buffer_size = 30  # ~1 second at 30 FPS
        
        # Vocabulary (placeholder)
        self.gloss_vocab = {i: f'gloss_{i}' for i in range(1000)}
        self.english_vocab = {i: f'word_{i}' for i in range(5000)}
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame and return results
        
        Args:
            frame: (H, W, 3) BGR image
        
        Returns:
            results: dict with 'gloss', 'english', 'latency_ms'
        """
        start_time = time.time()
        
        # Extract landmarks
        landmarks = self.landmark_extractor.extract_landmarks(frame)
        
        # Add to buffer
        self.landmark_buffer['hand'].append(landmarks['hand'])
        self.landmark_buffer['face'].append(landmarks['face'])
        self.landmark_buffer['pose'].append(landmarks['pose'])
        
        # Keep buffer size limited
        if len(self.landmark_buffer['hand']) > self.max_buffer_size:
            self.landmark_buffer['hand'].pop(0)
            self.landmark_buffer['face'].pop(0)
            self.landmark_buffer['pose'].pop(0)
        
        results = {
            'gloss': None,
            'english': None,
            'latency_ms': 0.0,
            'frame_count': len(self.landmark_buffer['hand'])
        }
        
        # Only infer when we have enough frames
        if len(self.landmark_buffer['hand']) >= 5:
            try:
                # Prepare batch
                hand_tensor = torch.from_numpy(np.array(self.landmark_buffer['hand'])).float().to(self.device)
                face_tensor = torch.from_numpy(np.array(self.landmark_buffer['face'])).float().to(self.device)
                pose_tensor = torch.from_numpy(np.array(self.landmark_buffer['pose'])).float().to(self.device)
                
                # Add batch dimension
                hand_tensor = hand_tensor.unsqueeze(0)
                face_tensor = face_tensor.unsqueeze(0)
                pose_tensor = pose_tensor.unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    # Recognize gloss
                    gloss_ids = self.model.recognize_gloss(hand_tensor, face_tensor, pose_tensor)
                    gloss_sequence = [self.gloss_vocab.get(int(id), 'UNK') for id in gloss_ids[0]]
                    
                    # Translate to English
                    english_ids = self.model.translate_gloss(gloss_ids)
                    english_sequence = [self.english_vocab.get(int(id), 'UNK') for id in english_ids[0]]
                
                results['gloss'] = ' '.join(filter(lambda x: x != 'UNK', gloss_sequence))
                results['english'] = ' '.join(filter(lambda x: x != 'UNK', english_sequence))
            
            except Exception as e:
                results['error'] = str(e)
        
        latency = (time.time() - start_time) * 1000
        results['latency_ms'] = latency
        
        return results
    
    def run_webcam(self, camera_id: int = 0):
        """Run real-time inference from webcam"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time inference. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results
            cv2.putText(frame, f"Frames: {results['frame_count']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Latency: {results['latency_ms']:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results['gloss']:
                cv2.putText(frame, f"Gloss: {results['gloss']}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if results['english']:
                cv2.putText(frame, f"Caption: {results['english']}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display
            cv2.imshow('ASL Recognition', frame)
            
            # Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video_file(self, video_path: str, output_path: str = None):
        """Process video file and optionally save output"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        results_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame)
            results_list.append(results)
            
            # Draw results
            if results['gloss']:
                cv2.putText(frame, f"Gloss: {results['gloss']}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if results['english']:
                cv2.putText(frame, f"Caption: {results['english']}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, Avg latency: {np.mean([r['latency_ms'] for r in results_list[-30:]]):.1f}ms")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Completed processing {frame_count} frames")
        print(f"Average latency: {np.mean([r['latency_ms'] for r in results_list]):.1f}ms")
        
        return results_list
