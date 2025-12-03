"""
PyTorch Dataset for REMI Music Generation
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from glob import glob
from tqdm import tqdm

from vocab import Pop1K7Vocab
from midi_processor import midi_to_remi_events, extract_bar_segments, transpose_sequence


class REMIMusicDataset(Dataset):
    """Dataset for REMI music sequences"""

    def __init__(
        self,
        sequences: List[List[int]],
        vocab: Pop1K7Vocab,
        max_length: int = 2048,
        augment_online: bool = False
    ):
        """
        Args:
            sequences: List of token sequences
            vocab: Vocabulary object
            max_length: Maximum sequence length
            augment_online: Whether to apply online augmentation
        """
        self.sequences = sequences
        self.vocab = vocab
        self.max_length = max_length
        self.augment_online = augment_online

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx].copy()

        # Online augmentation (tempo variation via random position shift)
        if self.augment_online and np.random.random() < 0.3:
            tokens = self._apply_online_augmentation(tokens)

        # Add BOS token at start
        tokens = [self.vocab.bos_token_id] + tokens

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create input/target pairs (input is tokens[:-1], target is tokens[1:])
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        # Pad sequences
        input_length = len(input_ids)
        padding_length = self.max_length - 1 - input_length

        if padding_length > 0:
            input_ids = input_ids + [self.vocab.pad_token_id] * padding_length
            target_ids = target_ids + [self.vocab.pad_token_id] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * input_length + [0] * max(0, padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def _apply_online_augmentation(self, tokens: List[int]) -> List[int]:
        """Apply online augmentation to token sequence"""
        # Slight velocity variation
        result = []
        for token in tokens:
            event = self.vocab.idx2event.get(token, '')
            if event.startswith('Note_Velocity_') and np.random.random() < 0.2:
                try:
                    velocity = int(event.split('_')[2])
                    # Random velocity shift
                    new_velocity = velocity + np.random.choice([-4, -2, 2, 4])
                    new_velocity = max(40, min(86, new_velocity))
                    new_velocity = 40 + round((new_velocity - 40) / 2) * 2
                    new_event = f'Note_Velocity_{new_velocity}'
                    new_token = self.vocab.event2idx.get(new_event, token)
                    result.append(new_token)
                except (ValueError, IndexError):
                    result.append(token)
            else:
                result.append(token)
        return result


def prepare_data_from_midi(
    midi_dir: str,
    vocab: Pop1K7Vocab,
    cache_path: str,
    target_bars: int = 32,
    augment: bool = True,
    force_rebuild: bool = False
) -> List[List[int]]:
    """
    Prepare training data from MIDI files

    Args:
        midi_dir: Directory containing MIDI files
        vocab: Vocabulary object
        cache_path: Path to cache processed data
        target_bars: Target number of bars per sequence
        augment: Whether to apply data augmentation
        force_rebuild: Force rebuild even if cache exists

    Returns:
        List of token sequences
    """
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['sequences']

    print(f"Processing MIDI files from {midi_dir}")

    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(glob(os.path.join(midi_dir, '**', ext), recursive=True))

    print(f"Found {len(midi_files)} MIDI files")

    all_sequences = []

    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        events = midi_to_remi_events(midi_path, max_bars=64)

        if len(events) < 100:  # Skip very short pieces
            continue

        # Convert events to tokens
        tokens = vocab.encode(events)

        # Extract 32-bar segments
        segments = extract_bar_segments(tokens, vocab, target_bars=target_bars, overlap_bars=8)

        for seg_tokens in segments:
            all_sequences.append(seg_tokens)

            # Data augmentation - pitch transposition
            if augment:
                for transpose in [-3, -1, 1, 3]:  # 4 transpositions
                    aug_tokens = transpose_sequence(seg_tokens, vocab, transpose)
                    if aug_tokens:
                        all_sequences.append(aug_tokens)

    print(f"Total sequences after processing: {len(all_sequences)}")

    # Cache the processed data
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'sequences': all_sequences}, f)

    return all_sequences


def create_dataloaders(
    sequences: List[List[int]],
    vocab: Pop1K7Vocab,
    batch_size: int = 32,
    max_length: int = 2048,
    train_split: float = 0.95,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        sequences: List of token sequences
        vocab: Vocabulary object
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Shuffle and split
    indices = np.random.permutation(len(sequences))
    split_idx = int(len(sequences) * train_split)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    # Create datasets
    train_dataset = REMIMusicDataset(
        train_sequences, vocab, max_length=max_length, augment_online=True
    )
    val_dataset = REMIMusicDataset(
        val_sequences, vocab, max_length=max_length, augment_online=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset creation
    vocab_path = '/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl'
    vocab = Pop1K7Vocab(vocab_path)

    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"PAD token: {vocab.pad_token_id}")
    print(f"BOS token: {vocab.bos_token_id}")
    print(f"EOS token: {vocab.eos_token_id}")

    # Test with small sample
    midi_dir = '/home/htiuser02/music/music/Pop1K7/Pop1K7/midi_analyzed/'
    cache_path = '/home/htiuser02/music/music/unconditional_generation/outputs/processed_data.pkl'

    sequences = prepare_data_from_midi(
        midi_dir, vocab, cache_path,
        target_bars=32, augment=True, force_rebuild=True
    )

    print(f"Number of sequences: {len(sequences)}")
    if sequences:
        print(f"First sequence length: {len(sequences[0])}")
        print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f}")
