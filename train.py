import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from pipeline import ASLRecognitionPipeline
from data_extractor import LandmarkDataset, LandmarkCollator


class Trainer:
    """Training loop for ASL recognition pipeline"""
    
    def __init__(self, model: ASLRecognitionPipeline, device: torch.device,
                 checkpoint_dir: str = './checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Metrics
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc='Training'):
            # Move to device
            hand = batch['landmarks'][:, :, :126].to(self.device)
            face = batch['landmarks'][:, :, 126:126+1404].to(self.device)
            pose = batch['landmarks'][:, :, 126+1404:].to(self.device)
            
            # For now, use dummy targets (would be replaced with actual data)
            batch_size = hand.shape[0]
            seq_len = hand.shape[1]
            
            gloss_targets = batch['gloss_ids'].unsqueeze(1).to(self.device)
            
            # Forward pass
            output = self.model(hand, face, pose, gloss_targets=gloss_targets)
            loss = output.get('rnnt_loss', torch.tensor(0.0, device=self.device))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                hand = batch['landmarks'][:, :, :126].to(self.device)
                face = batch['landmarks'][:, :, 126:126+1404].to(self.device)
                pose = batch['landmarks'][:, :, 126+1404:].to(self.device)
                
                batch_size = hand.shape[0]
                seq_len = hand.shape[1]
                
                gloss_targets = batch['gloss_ids'].unsqueeze(1).to(self.device)
                
                output = self.model(hand, face, pose, gloss_targets=gloss_targets)
                loss = output.get('rnnt_loss', torch.tensor(0.0, device=self.device))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 20, patience: int = 5):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_history.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_history.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Step scheduler
            self.scheduler.step(val_loss)
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print("✓ Saved best model")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        self.save_metrics()
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        filename = 'best.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        path = self.checkpoint_dir / filename
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        return checkpoint['epoch']
    
    def save_metrics(self):
        """Save training metrics"""
        metrics = {
            'train_loss': self.train_history,
            'val_loss': self.val_history,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_path = self.checkpoint_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


def main():
    """Main training script"""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset config
    import json
    with open('data/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    vocab_size = dataset_config['vocab_size']
    print(f"Loaded dataset config: {vocab_size} glosses")
    
    # Create model
    model = ASLRecognitionPipeline(vocab_size=vocab_size, english_vocab_size=5000)
    print(f"Created model with vocab_size={vocab_size}")
    
    # Create datasets with real data from metadata.json
    print("Loading datasets from data/metadata.json...")
    train_dataset = LandmarkDataset(data_dir='data/train', split='train', metadata_file='data/metadata.json')
    val_dataset = LandmarkDataset(data_dir='data/val', split='val', metadata_file='data/metadata.json')
    
    print(f"Train dataset: {len(train_dataset)} videos")
    print(f"Val dataset: {len(val_dataset)} videos")
    
    collator = LandmarkCollator()
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collator, 
                             num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collator, 
                           num_workers=0, shuffle=False)
    
    # Create trainer
    trainer = Trainer(model, device, checkpoint_dir='./checkpoints')
    
    # Train
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=20, patience=5)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
