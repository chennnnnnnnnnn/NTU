"""
Training Script for Music GPT-2 Model
Train for 120 epochs, save checkpoints at 100, 110, 120
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
# Using torch.amp instead of torch.cuda.amp (deprecated)
from tqdm import tqdm

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import Pop1K7Vocab
from dataset import prepare_data_from_midi, create_dataloaders
from model import create_model, MusicGPT2


def train_epoch(
    model: MusicGPT2,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    epoch: int,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1
) -> float:
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / gradient_accumulation_steps
            loss.backward()

        accumulated_loss += loss.item() * gradient_accumulation_steps

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            total_loss += accumulated_loss
            num_batches += 1
            accumulated_loss = 0

        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss/max(1, num_batches):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    # Handle remaining accumulated gradients
    if accumulated_loss > 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: MusicGPT2,
    val_loader,
    device: str,
    use_amp: bool = True
) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(
    model: MusicGPT2,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: str
):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}')
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_path)

    # Save training state
    state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(state, os.path.join(checkpoint_path, 'training_state.pt'))

    print(f"Checkpoint saved at {checkpoint_path}")


def train(
    midi_dir: str,
    output_dir: str,
    vocab_path: str,
    num_epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    warmup_epochs: int = 5,
    n_layer: int = 12,
    n_embd: int = 512,
    n_head: int = 8,
    max_length: int = 2048,
    checkpoint_epochs: list = [80, 100, 120],
    use_amp: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    resume_from: str = None
):
    """
    Main training function

    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Output directory for checkpoints
        vocab_path: Path to vocabulary file
        num_epochs: Total number of epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_epochs: Number of warmup epochs
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        n_head: Number of attention heads
        max_length: Maximum sequence length
        checkpoint_epochs: Epochs at which to save checkpoints
        use_amp: Use automatic mixed precision
        num_workers: Number of dataloader workers
        seed: Random seed
        resume_from: Path to checkpoint to resume from
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Pop1K7Vocab(vocab_path)
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Prepare data
    cache_path = os.path.join(output_dir, 'processed_data.pkl')
    print("Preparing training data...")
    sequences = prepare_data_from_midi(
        midi_dir, vocab, cache_path,
        target_bars=32, augment=True
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        sequences, vocab,
        batch_size=batch_size,
        max_length=max_length,
        train_split=0.95,
        num_workers=num_workers
    )

    # Create model
    print("Creating model...")
    model = create_model(
        vocab_size=vocab.vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        max_length=max_length,
        pad_token_id=vocab.pad_token_id,
        bos_token_id=vocab.bos_token_id,
        eos_token_id=vocab.eos_token_id,
        dropout=0.1
    )
    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Create scheduler
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )

    # Create gradient scaler for AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Gradient accumulation settings
    gradient_accumulation_steps = 4  # Effective batch size = batch_size * 4

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        model = MusicGPT2.from_pretrained(resume_from)
        model = model.to(device)

        state_path = os.path.join(resume_from, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            start_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}")
    print(f"Total epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoints will be saved at epochs: {checkpoint_epochs}")
    print("-" * 50)

    # Save training config
    config = {
        'midi_dir': midi_dir,
        'vocab_path': vocab_path,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'max_length': max_length,
        'vocab_size': vocab.vocab_size,
        'num_sequences': len(sequences),
        'checkpoint_epochs': checkpoint_epochs,
        'start_time': datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, use_amp, gradient_accumulation_steps
        )

        # Validate
        val_loss = validate(model, val_loader, device, use_amp)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint at specified epochs
        if epoch in checkpoint_epochs:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss, val_loss,
                checkpoint_dir
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, 'best_model')
            model.save_pretrained(best_path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved at: {checkpoint_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Music GPT-2 Model')

    parser.add_argument('--midi_dir', type=str,
                        default='/home/htiuser02/music/music/Pop1K7/Pop1K7/midi_analyzed/',
                        help='Directory containing MIDI files')
    parser.add_argument('--output_dir', type=str,
                        default='/home/htiuser02/music/music/unconditional_generation/outputs/',
                        help='Output directory')
    parser.add_argument('--vocab_path', type=str,
                        default='/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl',
                        help='Path to vocabulary file')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Model parameters
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=2048)

    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()

    train(
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        max_length=args.max_length,
        checkpoint_epochs=[80, 100, 120],
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        seed=args.seed,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
