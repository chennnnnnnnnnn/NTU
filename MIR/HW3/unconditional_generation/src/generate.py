"""
Generation Script for Music GPT-2 Model
Generate MIDI files with different sampling configurations
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import Pop1K7Vocab
from model import MusicGPT2
from midi_processor import remi_events_to_midi


# Sampling configurations
SAMPLING_CONFIGS = [
    {"name": "config1", "top_k": 5, "temperature": 0.9},
    {"name": "config2", "top_k": 8, "temperature": 1.0},
    {"name": "config3", "top_k": 10, "temperature": 1.15},
]


def generate_samples(
    model: MusicGPT2,
    vocab: Pop1K7Vocab,
    num_samples: int,
    num_bars: int = 32,
    max_length: int = 10000,
    top_k: int = 10,
    temperature: float = 1.0,
    device: str = 'cuda',
    show_progress: bool = True
) -> List[List[int]]:
    """
    Generate multiple music samples

    Args:
        model: Trained MusicGPT2 model
        vocab: Vocabulary object
        num_samples: Number of samples to generate
        num_bars: Target number of bars per sample
        max_length: Maximum sequence length
        top_k: Top-k sampling parameter
        temperature: Sampling temperature
        device: Device to use
        show_progress: Show progress bar

    Returns:
        List of generated token sequences
    """
    model.eval()
    samples = []

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating (k={top_k}, t={temperature})")

    for _ in iterator:
        # Generate sequence
        with torch.no_grad():
            generated = model.generate(
                input_ids=None,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_bars=num_bars,
                bar_token_id=vocab.bar_token,
                eos_token_id=vocab.eos_token_id,
                device=device
            )

        # Convert to list and remove BOS token
        tokens = generated[0].cpu().tolist()
        if tokens[0] == vocab.bos_token_id:
            tokens = tokens[1:]

        samples.append(tokens)

    return samples


def ensure_32_continuous_bars(events: List[str], target_bars: int = 32) -> List[str]:
    """
    Post-process events to ensure exactly target_bars bars, all with notes.
    Keeps only bars that have notes, then takes first target_bars.

    Args:
        events: List of REMI events
        target_bars: Target number of bars (default 32)

    Returns:
        Processed events with exactly target_bars continuous bars
    """
    import random

    # Find bar boundaries
    bar_indices = [i for i, e in enumerate(events) if e == 'Bar_None']

    if not bar_indices:
        return events

    # Parse events into bars
    all_bars = []
    for i, bar_idx in enumerate(bar_indices):
        if i + 1 < len(bar_indices):
            next_bar_idx = bar_indices[i + 1]
        else:
            next_bar_idx = len(events)
        bar_content = events[bar_idx:next_bar_idx]
        has_notes = any(e.startswith('Note_Pitch_') for e in bar_content)
        all_bars.append({
            'content': bar_content,
            'has_notes': has_notes
        })

    # Collect bars with notes
    bars_with_notes = [b['content'] for b in all_bars if b['has_notes']]

    # Take first target_bars bars with notes
    result_bars = bars_with_notes[:target_bars]

    # If we don't have enough bars, duplicate existing ones
    while len(result_bars) < target_bars and len(result_bars) > 0:
        source_bar = random.choice(result_bars)
        result_bars.append(list(source_bar))

    # Reconstruct events - exactly target_bars bars
    new_events = []
    for bar_content in result_bars:
        new_events.extend(bar_content)

    return new_events


def tokens_to_midi(
    tokens: List[int],
    vocab: Pop1K7Vocab,
    output_path: str,
    ensure_32_bars: bool = True
) -> bool:
    """
    Convert tokens to MIDI file

    Args:
        tokens: Token sequence
        vocab: Vocabulary object
        output_path: Output MIDI file path
        ensure_32_bars: Whether to ensure exactly 32 continuous bars

    Returns:
        True if successful
    """
    # Convert tokens to events
    events = vocab.decode(tokens)

    # Filter out special tokens
    events = [e for e in events if not e.startswith('<')]

    # Ensure 32 continuous bars with notes
    if ensure_32_bars:
        events = ensure_32_continuous_bars(events, target_bars=32)

    # Convert to MIDI
    return remi_events_to_midi(events, output_path)


def generate_for_all_configs(
    model: MusicGPT2,
    vocab: Pop1K7Vocab,
    output_dir: str,
    num_samples: int = 20,
    num_bars: int = 32,
    max_length: int = 10000,
    device: str = 'cuda'
) -> Dict[str, List[str]]:
    """
    Generate samples for all sampling configurations
    Generates and saves one sample at a time for immediate verification

    Args:
        model: Trained model
        vocab: Vocabulary object
        output_dir: Output directory
        num_samples: Number of samples per configuration
        num_bars: Number of bars to generate
        device: Device to use

    Returns:
        Dictionary mapping config name to list of MIDI file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for config in SAMPLING_CONFIGS:
        config_name = config['name']
        top_k = config['top_k']
        temperature = config['temperature']

        print(f"\nGenerating for {config_name} (top_k={top_k}, temperature={temperature})")

        # Create config output directory
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Generate samples ONE AT A TIME
        midi_paths = []
        all_tokens = []

        for i in range(num_samples):
            print(f"  Generating sample {i+1}/{num_samples}...")

            # Generate single sample (generate extra bars as buffer for post-processing)
            generate_bars = num_bars + 5  # Extra bars to ensure we have enough with notes
            with torch.no_grad():
                generated = model.generate(
                    input_ids=None,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    num_bars=generate_bars,
                    bar_token_id=vocab.bar_token,
                    eos_token_id=vocab.eos_token_id,
                    device=device
                )

            # Convert to list and remove BOS token
            tokens = generated[0].cpu().tolist()
            if tokens[0] == vocab.bos_token_id:
                tokens = tokens[1:]

            all_tokens.append(tokens)

            # Immediately save as MIDI file
            midi_path = os.path.join(config_dir, f'sample_{i:03d}.mid')
            success = tokens_to_midi(tokens, vocab, midi_path)

            if success:
                midi_paths.append(midi_path)
                # Count bars for verification
                from midi_processor import midi_to_remi_events
                events = midi_to_remi_events(midi_path)
                bar_count = sum(1 for e in events if e.startswith('Bar_'))
                status = "✓" if bar_count == 32 else f"✗ ({bar_count} bars)"
                print(f"    {status} Saved: {os.path.basename(midi_path)}")
            else:
                print(f"    ✗ Warning: Failed to save {midi_path}")

        results[config_name] = midi_paths
        print(f"  Total: {len(midi_paths)} MIDI files generated")

        # Save token sequences for analysis
        tokens_path = os.path.join(config_dir, 'token_sequences.json')
        with open(tokens_path, 'w') as f:
            json.dump({'samples': all_tokens}, f)

    return results


def generate_from_checkpoint(
    checkpoint_path: str,
    vocab_path: str,
    output_dir: str,
    num_samples: int = 20,
    num_bars: int = 32,
    max_length: int = 10000,
    config_name: str = None,
    top_k: int = None,
    temperature: float = None,
) -> Dict[str, List[str]]:
    """
    Generate samples from a specific checkpoint

    Args:
        checkpoint_path: Path to model checkpoint
        vocab_path: Path to vocabulary file
        output_dir: Output directory
        num_samples: Number of samples
        num_bars: Number of bars
        config_name: Specific config to use (None for all configs)
        top_k: Override top_k (only if config_name is None)
        temperature: Override temperature (only if config_name is None)

    Returns:
        Results dictionary
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Pop1K7Vocab(vocab_path)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = MusicGPT2.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Determine top_k and temperature values
    if top_k is not None and temperature is not None:
        # Use command-line arguments directly
        use_top_k = top_k
        use_temperature = temperature
        config_label = f"custom_k{top_k}_t{temperature}"
        print(f"\nGenerating with custom params (top_k={use_top_k}, temperature={use_temperature})")
    elif config_name:
        # Use specific config
        config = next((c for c in SAMPLING_CONFIGS if c['name'] == config_name), None)
        if config is None:
            raise ValueError(f"Unknown config: {config_name}")
        use_top_k = config['top_k']
        use_temperature = config['temperature']
        config_label = config_name
        print(f"\nGenerating for {config_name} (top_k={use_top_k}, temperature={use_temperature})")
    else:
        # Generate for all configs
        return generate_for_all_configs(
            model, vocab, output_dir,
            num_samples=num_samples,
            num_bars=num_bars,
            max_length=max_length,
            device=device
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate samples ONE AT A TIME and save immediately
    midi_paths = []
    for i in range(num_samples):
        print(f"  Generating sample {i+1}/{num_samples}...", flush=True)

        # Generate single sample (generate extra bars as buffer for post-processing)
        generate_bars = num_bars + 5  # Extra bars to ensure we have enough with notes
        with torch.no_grad():
            generated = model.generate(
                input_ids=None,
                max_length=max_length,
                temperature=use_temperature,
                top_k=use_top_k,
                num_bars=generate_bars,
                bar_token_id=vocab.bar_token,
                eos_token_id=vocab.eos_token_id,
                device=device
            )

        # Convert to list and remove BOS token
        tokens = generated[0].cpu().tolist()
        if tokens[0] == vocab.bos_token_id:
            tokens = tokens[1:]

        # Immediately save as MIDI file
        midi_path = os.path.join(output_dir, f'sample_{i:03d}.mid')
        success = tokens_to_midi(tokens, vocab, midi_path)

        if success:
            midi_paths.append(midi_path)
            print(f"    ✓ Saved: {os.path.basename(midi_path)}", flush=True)
        else:
            print(f"    ✗ Warning: Failed to save {midi_path}", flush=True)

    return {config_label: midi_paths}


def main():
    parser = argparse.ArgumentParser(description='Generate Music with GPT-2')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str,
                        default='/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl',
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str,
                        default='/home/htiuser02/music/music/unconditional_generation/generated_midi/',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate')
    parser.add_argument('--num_bars', type=int, default=32,
                        help='Number of bars to generate')
    parser.add_argument('--max_length', type=int, default=10000,
                        help='Maximum sequence length (shorter = faster generation)')
    parser.add_argument('--config', type=str, default=None,
                        choices=['config1', 'config2', 'config3'],
                        help='Specific sampling config to use')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling parameter (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (overrides config)')

    args = parser.parse_args()

    results = generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_bars=args.num_bars,
        max_length=args.max_length,
        config_name=args.config,
        top_k=args.top_k,
        temperature=args.temperature
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Generation Summary")
    print("=" * 50)
    for config_name, paths in results.items():
        print(f"{config_name}: {len(paths)} files")


if __name__ == '__main__':
    main()
