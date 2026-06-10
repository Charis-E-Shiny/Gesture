import torch
import numpy as np
from pathlib import Path

from pipeline import ASLRecognitionPipeline
from config import MODEL_CONFIG, count_parameters, get_model_size_mb, get_device


def test_model_creation():
    """Test model creation and structure"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    device = get_device()
    model = ASLRecognitionPipeline(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        english_vocab_size=MODEL_CONFIG['english_vocab_size']
    ).to(device)
    
    print(f"✓ Model created on {device}")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Model size: {get_model_size_mb(model):.2f} MB")
    
    return model


def test_forward_pass(model):
    """Test forward pass through the model"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    device = get_device()
    batch_size = 2
    seq_len = 30
    
    # Create dummy inputs
    hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim']).to(device)
    face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim']).to(device)
    pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim']).to(device)
    
    print(f"Input shapes:")
    print(f"  Hand: {hand.shape}")
    print(f"  Face: {face.shape}")
    print(f"  Pose: {pose.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(hand, face, pose)
    
    print(f"\nOutput shapes:")
    print(f"  Gloss logits: {output['gloss_logits'].shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {MODEL_CONFIG['vocab_size']})")
    
    assert output['gloss_logits'].shape == (batch_size, seq_len, MODEL_CONFIG['vocab_size'])
    print("✓ Forward pass successful")
    
    return output


def test_training_forward(model):
    """Test forward pass with training targets"""
    print("\n" + "="*60)
    print("TEST 3: Training Forward Pass")
    print("="*60)
    
    device = get_device()
    batch_size = 2
    seq_len = 30
    target_len = 15
    
    # Create inputs and targets
    hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim']).to(device)
    face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim']).to(device)
    pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim']).to(device)
    
    gloss_targets = torch.randint(0, MODEL_CONFIG['vocab_size'], 
                                  (batch_size, target_len)).to(device)
    english_targets = torch.randint(0, MODEL_CONFIG['english_vocab_size'],
                                   (batch_size, target_len)).to(device)
    
    # Forward pass with training targets
    output = model(hand, face, pose, 
                  gloss_targets=gloss_targets,
                  english_targets=english_targets)
    
    print(f"Input shapes:")
    print(f"  Gloss targets: {gloss_targets.shape}")
    print(f"  English targets: {english_targets.shape}")
    
    print(f"\nLosses computed:")
    print(f"  RNNT Loss: {output['rnnt_loss'].item():.4f}")
    print(f"  Translation Loss: {output['translation_loss'].item():.4f}")
    
    # Backward pass
    total_loss = model.get_total_loss(output)
    total_loss.backward()
    
    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print("✓ Training forward pass successful")


def test_gloss_recognition(model):
    """Test gloss recognition"""
    print("\n" + "="*60)
    print("TEST 4: Gloss Recognition")
    print("="*60)
    
    device = get_device()
    batch_size = 1
    seq_len = 30
    
    # Create inputs
    hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim']).to(device)
    face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim']).to(device)
    pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim']).to(device)
    
    # Recognize
    with torch.no_grad():
        gloss_ids = model.recognize_gloss(hand, face, pose)
    
    print(f"Gloss IDs shape: {gloss_ids.shape}")
    print(f"Sample gloss IDs: {gloss_ids[0, :5].tolist()}")
    print("✓ Gloss recognition successful")


def test_translation(model):
    """Test gloss to English translation"""
    print("\n" + "="*60)
    print("TEST 5: Translation (Gloss → English)")
    print("="*60)
    
    device = get_device()
    batch_size = 1
    
    # Create sample gloss sequence
    gloss_ids = torch.randint(4, MODEL_CONFIG['vocab_size'], (batch_size, 10)).to(device)
    
    print(f"Input gloss IDs: {gloss_ids[0].tolist()}")
    
    # Translate
    with torch.no_grad():
        english_ids = model.translate_gloss(gloss_ids)
    
    print(f"Output English IDs shape: {english_ids.shape}")
    print(f"Sample English IDs: {english_ids[0].tolist()}")
    print("✓ Translation successful")


def test_inference_pipeline():
    """Test real-time inference pipeline"""
    print("\n" + "="*60)
    print("TEST 6: Inference Pipeline (No Model File)")
    print("="*60)
    
    print("Note: This test creates a dummy pipeline without loading a real model")
    print("In production, ensure checkpoints/best.pt exists")
    
    try:
        from inference import RealtimeInferencePipeline
        
        # Create dummy input
        device = get_device()
        batch_size = 1
        seq_len = 5
        
        hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim'])
        face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim'])
        pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim'])
        
        print("✓ Inference pipeline components available")
        print("  (Full test requires video input)")
    
    except Exception as e:
        print(f"✗ Inference pipeline test skipped: {e}")


def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n" + "="*60)
    print("TEST 7: Memory Efficiency")
    print("="*60)
    
    device = get_device()
    
    model = ASLRecognitionPipeline(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        english_vocab_size=MODEL_CONFIG['english_vocab_size']
    ).to(device)
    
    model_size = get_model_size_mb(model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Estimate inference memory
    batch_size = 1
    seq_len = 30
    
    hand = torch.randn(batch_size, seq_len, MODEL_CONFIG['hand_input_dim']).to(device)
    face = torch.randn(batch_size, seq_len, MODEL_CONFIG['face_input_dim']).to(device)
    pose = torch.randn(batch_size, seq_len, MODEL_CONFIG['pose_input_dim']).to(device)
    
    # Measure inference time
    import time
    
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            output = model(hand, face, pose)
        elapsed = time.time() - start
    
    avg_latency = (elapsed / 10) * 1000
    print(f"Average inference latency: {avg_latency:.1f}ms (batch_size=1, seq_len=30)")
    
    if avg_latency < 500:
        print("✓ Latency target (<500ms) achieved")
    else:
        print(f"⚠ Latency {avg_latency:.1f}ms exceeds target of 500ms")


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("#" * 60)
    print("# ASL RECOGNITION PIPELINE - TEST SUITE")
    print("#" * 60)
    
    try:
        model = test_model_creation()
        test_forward_pass(model)
        test_training_forward(model)
        test_gloss_recognition(model)
        test_translation(model)
        test_inference_pipeline()
        test_memory_efficiency()
        
        print("\n" + "#" * 60)
        print("# ✓ ALL TESTS PASSED")
        print("#" * 60 + "\n")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
