"""
Example usage of the ASL Recognition & Translation Pipeline

This script demonstrates how to:
1. Create and use the complete pipeline
2. Train the model
3. Run real-time inference
4. Start the API server
"""

import torch
import cv2
from pathlib import Path

from pipeline import ASLRecognitionPipeline
from inference import RealtimeInferencePipeline
from data_extractor import MediaPipeLandmarkExtractor
from config import MODEL_CONFIG, TRAINING_CONFIG, get_device


# ============ Example 1: Basic Inference ============

def example_basic_inference():
    """Example 1: Basic inference on dummy data"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Inference")
    print("="*60)
    
    device = get_device()
    
    # Create model
    model = ASLRecognitionPipeline(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        english_vocab_size=MODEL_CONFIG['english_vocab_size']
    ).to(device)
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 30
    
    hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim']).to(device)
    face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim']).to(device)
    pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim']).to(device)
    
    # Forward pass
    with torch.no_grad():
        # Recognize gloss
        gloss_ids = model.recognize_gloss(hand, face, pose)
        print(f"Recognized gloss IDs: {gloss_ids[0].tolist()}")
        
        # Translate to English
        english_ids = model.translate_gloss(gloss_ids)
        print(f"English IDs: {english_ids[0].tolist()}")


# ============ Example 2: Processing Video File ============

def example_process_video():
    """Example 2: Process a video file"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Process Video File")
    print("="*60)
    
    # Path to video file
    video_path = "path/to/asl_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("To use this example, provide a real video file")
        return
    
    # Initialize inference pipeline
    pipeline = RealtimeInferencePipeline(
        model_path='./checkpoints/best.pt'
    )
    
    # Process video
    results = pipeline.process_video_file(
        video_path,
        output_path="asl_video_with_captions.mp4"
    )
    
    # Print results
    print(f"Processed {len(results)} frames")
    for i, result in enumerate(results[:5]):
        print(f"Frame {i}: {result.get('english', 'N/A')}")


# ============ Example 3: Real-time Webcam ============

def example_webcam():
    """Example 3: Real-time webcam inference"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Real-time Webcam Inference")
    print("="*60)
    
    print("Starting webcam inference...")
    print("Press 'q' to quit\n")
    
    # Initialize pipeline
    try:
        pipeline = RealtimeInferencePipeline(
            model_path='./checkpoints/best.pt'
        )
        
        # Run webcam
        pipeline.run_webcam(camera_id=0)
    
    except FileNotFoundError:
        print("Error: Model checkpoint not found at ./checkpoints/best.pt")
        print("Train the model first using: python train.py")


# ============ Example 4: Landmark Extraction ============

def example_landmark_extraction():
    """Example 4: Extract landmarks from video"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Landmark Extraction")
    print("="*60)
    
    video_path = "path/to/asl_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Creating dummy example instead...")
        
        # Create dummy video
        return
    
    # Extract landmarks
    extractor = MediaPipeLandmarkExtractor()
    hand_lm, face_lm, pose_lm = extractor.process_video(video_path, max_frames=30)
    
    print(f"Hand landmarks shape: {hand_lm.shape}")
    print(f"Face landmarks shape: {face_lm.shape}")
    print(f"Pose landmarks shape: {pose_lm.shape}")
    
    # Show sample
    print(f"\nSample hand landmarks (frame 0):\n{hand_lm[0][:10]}")


# ============ Example 5: Using with FastAPI ============

def example_api_usage():
    """Example 5: Using the FastAPI endpoints"""
    print("\n" + "="*60)
    print("EXAMPLE 5: FastAPI Server Usage")
    print("="*60)
    
    print("To start the API server, run:")
    print("  python app.py")
    print("")
    print("Then use the API endpoints:")
    print("  GET http://localhost:8000/health")
    print("  POST http://localhost:8000/infer_frame")
    print("  WebSocket ws://localhost:8000/ws")
    print("  POST http://localhost:8000/upload_video")
    print("")
    print("Python example:")
    
    example_code = '''
import requests
import cv2
import base64

# Read frame
frame = cv2.imread('frame.jpg')
_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode()

# Send to API
response = requests.post(
    'http://localhost:8000/infer_frame',
    json={'frame_base64': frame_base64, 'frame_id': 0}
)

print(response.json())
# Output: {
#     'gloss': 'HELLO WORLD',
#     'english': 'Hello world',
#     'latency_ms': 120.5,
#     'frame_count': 30
# }
    '''
    
    print(example_code)


# ============ Example 6: Custom Training ============

def example_custom_training():
    """Example 6: Training with custom data"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Training")
    print("="*60)
    
    print("To train the model with custom data:")
    print("")
    print("1. Prepare dataset:")
    print("   data/")
    print("   ├── train/gloss_name/video_1.mp4")
    print("   ├── val/gloss_name/video_2.mp4")
    print("   └── test/gloss_name/video_3.mp4")
    print("")
    print("2. Run training:")
    print("   python train.py")
    print("")
    print("3. Monitor with TensorBoard:")
    print("   tensorboard --logdir ./logs")


# ============ Example 7: Batch Processing ============

def example_batch_processing():
    """Example 7: Batch processing multiple videos"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Batch Processing")
    print("="*60)
    
    from pathlib import Path
    
    device = get_device()
    
    # Create model
    model = ASLRecognitionPipeline(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        english_vocab_size=MODEL_CONFIG['english_vocab_size']
    ).to(device)
    
    # Process multiple videos
    video_dir = Path('./data/test')
    video_files = list(video_dir.glob('**/*.mp4'))
    
    print(f"Found {len(video_files)} video files")
    
    results = []
    for video_file in video_files[:3]:  # Process first 3
        print(f"Processing: {video_file}")
        
        # Extract landmarks
        from data_extractor import MediaPipeLandmarkExtractor
        extractor = MediaPipeLandmarkExtractor()
        
        try:
            hand_lm, face_lm, pose_lm = extractor.process_video(str(video_file), max_frames=30)
            print(f"  Extracted {len(hand_lm)} frames")
            results.append({
                'video': str(video_file),
                'frames': len(hand_lm),
                'status': 'success'
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'video': str(video_file),
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\nProcessed {len(results)} videos")
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"Successful: {successful}")


# ============ Example 8: Model Analysis ============

def example_model_analysis():
    """Example 8: Analyze model architecture"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Model Analysis")
    print("="*60)
    
    from config import count_parameters, get_model_size_mb
    
    device = get_device()
    
    # Create model
    model = ASLRecognitionPipeline(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        english_vocab_size=MODEL_CONFIG['english_vocab_size']
    ).to(device)
    
    # Analysis
    print(f"Model Components:")
    print(f"  Hand Encoder: {model.hand_encoder}")
    print(f"  Face Encoder: {model.face_encoder}")
    print(f"  Pose Encoder: {model.pose_encoder}")
    print(f"  Fusion: {type(model.fusion).__name__}")
    print(f"  Conformer: {type(model.conformer).__name__}")
    print(f"  RNNT: {type(model.rnnt).__name__}")
    print(f"  Translator: {type(model.translator).__name__}")
    
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {count_parameters(model):,}")
    print(f"  Model Size: {get_model_size_mb(model):.2f} MB")
    
    print(f"\nLayer Breakdown:")
    total_params = 0
    for name, module in model.named_children():
        params = count_parameters(module)
        total_params += params
        print(f"  {name}: {params:,} parameters")


# ============ Main Menu ============

def main():
    """Run example menu"""
    print("\n")
    print("#" * 60)
    print("# ASL RECOGNITION PIPELINE - EXAMPLES")
    print("#" * 60)
    
    examples = {
        '1': ('Basic Inference', example_basic_inference),
        '2': ('Process Video File', example_process_video),
        '3': ('Real-time Webcam', example_webcam),
        '4': ('Landmark Extraction', example_landmark_extraction),
        '5': ('FastAPI Usage', example_api_usage),
        '6': ('Custom Training', example_custom_training),
        '7': ('Batch Processing', example_batch_processing),
        '8': ('Model Analysis', example_model_analysis),
        '0': ('Run All Examples', lambda: [func() for func in 
              [v[1] for v in examples.values() if v != examples['0']]])
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        if key != '0':
            print(f"  {key}. {name}")
    print(f"  0. Run All Examples")
    print(f"  q. Quit")
    
    while True:
        choice = input("\nSelect example (1-8, 0, or q): ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            break
        
        if choice in examples:
            try:
                examples[choice][1]()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
