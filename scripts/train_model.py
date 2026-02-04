#!/usr/bin/env python3
"""
Train TinyTransformer on bilingual corpus using next-token prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from pathlib import Path
import time
import json
import argparse
import uuid
from datetime import datetime
from tqdm import tqdm

# Import our model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.tiny_transformer import TinyTransformer

class TextDataset(Dataset):
    """Dataset for language modeling (next-token prediction)."""
    
    def __init__(self, text_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize all text
        print(f"📖 Loading text from {text_file}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize into chunks
        print("🔤 Tokenizing text...")
        all_tokens = self.tokenizer.encode(text, out_type=int)
        
        # Split into sequences of max_length
        self.sequences = []
        for i in range(0, len(all_tokens) - max_length - 1, max_length // 2):
            seq = all_tokens[i:i + max_length + 1]
            if len(seq) == max_length + 1:
                self.sequences.append(seq)
        
        print(f"   ✓ Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input: first max_length tokens
        # Target: shifted by 1 (predict next token)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Calculate loss (flatten for cross-entropy)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=0.7, device='cpu'):
    """Generate text sample to check model quality during training."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate if too long
            if input_ids.size(1) > 1000:
                input_ids = input_ids[:, -1000:]
            
            # Forward pass
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if EOS
            if next_token == tokenizer.eos_id():
                break
            
            # Append
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    
    # Decode
    text = tokenizer.decode(generated)
    return text


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train TinyTransformer on bilingual corpus')
    parser.add_argument('--vocab-size', type=int, default=8000, help='Vocabulary size')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--max-seq-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory for artifacts')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'vocab_size': args.vocab_size,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'max_seq_length': args.max_seq_length,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 5,  # Save checkpoint every N epochs
        'output_dir': args.output_dir,
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 TinyTransformer Training")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor(model_file='data/bilingual_8k.model')
    print(f"   ✓ Vocabulary size: {tokenizer.vocab_size()}")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    full_dataset = TextDataset('data/corpus_bilingual.txt', tokenizer, config['max_seq_length'])
    
    # Split into train/val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"   ✓ Training samples: {len(train_dataset)}")
    print(f"   ✓ Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, can increase for speed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n🧠 Initializing model...")
    model = TinyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward']
    ).to(config['device'])
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # Training loop
    print("\n🏋️ Starting training...")
    print("=" * 60)
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    start_timestamp = datetime.now().isoformat()
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            config['device'], epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config['device'])
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Time taken
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"\nEpoch {epoch}/{config['num_epochs']} - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })
        
        # Generate sample text
        if epoch % 5 == 0 or epoch == 1:
            print("\n  📝 Sample generation:")
            for prompt in ["Hello", "Hola"]:
                sample = generate_sample(
                    model, tokenizer, prompt, 
                    max_tokens=30, temperature=0.7, 
                    device=config['device']
                )
                print(f"    '{prompt}' → {sample}")
        
        # Save checkpoint
        if epoch % config['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % config['save_every'] == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"  💾 Saved checkpoint epoch {epoch}")
        
        print("-" * 60)
    
    # Save final model
    print("\n✅ Training complete!")
    end_timestamp = datetime.now().isoformat()
    
    final_checkpoint = {
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'training_history': training_history
    }
    final_model_path = output_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_model_path)
    print(f"   💾 Saved final model: {final_model_path}")
    print(f"   💾 Best model: {output_dir / 'best_model.pt'} (val_loss: {best_val_loss:.4f})")
    
    # Export training metrics to JSON
    training_metrics = {
        'run_id': run_id,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'configuration': {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                          for k, v in config.items()},
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_train_loss': training_history[-1]['train_loss'] if training_history else None,
            'final_val_loss': training_history[-1]['val_loss'] if training_history else None,
            'total_epochs': len(training_history),
        },
        'epoch_history': training_history,
    }
    
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"   📊 Exported training metrics: {metrics_path}")
    
    # Final samples
    print("\n🎭 Final generation samples:")
    test_prompts = [
        "Hello world",
        "The cat sat on the",
        "Hola mundo",
        "El gato"
    ]
    for prompt in test_prompts:
        sample = generate_sample(
            model, tokenizer, prompt,
            max_tokens=50, temperature=0.7,
            device=config['device']
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {sample}")


if __name__ == '__main__':
    main()
