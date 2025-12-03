"""
Quick Test Script: Test if model can generate 32 bars
"""

import os
import sys
import torch

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import Pop1K7Vocab
from model import create_model, MusicGPT2

# Configuration
VOCAB_PATH = '/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl'
OUTPUT_DIR = '/home/htiuser02/music/music/unconditional_generation/test_output'


def test_generation(checkpoint_path=None, num_bars=32):
    """
    Test generation with a model (random weights if no checkpoint)

    Args:
        checkpoint_path: Path to trained checkpoint (None for random model)
        num_bars: Target number of bars to generate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Pop1K7Vocab(VOCAB_PATH)
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Bar token ID: {vocab.bar_token}")
    print(f"BOS token ID: {vocab.bos_token_id}")
    print(f"EOS token ID: {vocab.eos_token_id}")

    # Create or load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = MusicGPT2.from_pretrained(checkpoint_path)
    else:
        print("Creating new model with random weights (for testing pipeline)")
        model = create_model(
            vocab_size=vocab.vocab_size,
            n_layer=12,
            n_embd=512,
            n_head=8,
            max_length=2048,
            pad_token_id=vocab.pad_token_id,
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            dropout=0.1
        )

    model = model.to(device)
    model.eval()

    # Test generation
    print(f"\n{'='*50}")
    print(f"Testing generation of {num_bars} bars...")
    print(f"{'='*50}")

    with torch.no_grad():
        generated = model.generate(
            input_ids=None,
            max_length=4096,  # Allow longer sequences for 32 bars
            temperature=1.0,
            top_k=10,
            num_bars=num_bars,
            bar_token_id=vocab.bar_token,
            eos_token_id=vocab.eos_token_id,
            device=device
        )

    # Analyze generated sequence
    tokens = generated[0].cpu().tolist()
    print(f"\nGenerated sequence length: {len(tokens)} tokens")

    # Remove BOS token
    if tokens[0] == vocab.bos_token_id:
        tokens = tokens[1:]

    # Count bars
    bar_count = sum(1 for t in tokens if t == vocab.bar_token)
    print(f"Number of Bar tokens: {bar_count}")

    # Decode to events
    events = vocab.decode(tokens)

    # Count different event types
    event_counts = {}
    for event in events:
        event_type = event.split('_')[0]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    print(f"\nEvent distribution:")
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")

    # Try to save as MIDI
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from midi_processor import remi_events_to_midi

    # Filter out special tokens
    filtered_events = [e for e in events if not e.startswith('<')]

    midi_path = os.path.join(OUTPUT_DIR, f'test_{num_bars}bars.mid')
    success = remi_events_to_midi(filtered_events, midi_path)

    if success:
        print(f"\n✓ Successfully saved MIDI to: {midi_path}")

        # Verify MIDI file
        try:
            import mido
            mid = mido.MidiFile(midi_path)
            print(f"  MIDI duration: {mid.length:.2f} seconds")
            print(f"  Number of tracks: {len(mid.tracks)}")
        except Exception as e:
            print(f"  Warning: Could not verify MIDI: {e}")
    else:
        print(f"\n✗ Failed to save MIDI")

    # Summary
    print(f"\n{'='*50}")
    print("GENERATION TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Target bars: {num_bars}")
    print(f"Generated bars: {bar_count}")
    print(f"Sequence length: {len(tokens)} tokens")
    print(f"Status: {'✓ SUCCESS' if bar_count >= num_bars else '○ PARTIAL (need more training)'}")

    return {
        'target_bars': num_bars,
        'generated_bars': bar_count,
        'sequence_length': len(tokens),
        'events': filtered_events,
        'success': bar_count >= num_bars
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test 32-bar generation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--num_bars', type=int, default=32,
                        help='Number of bars to generate')

    args = parser.parse_args()

    # Check for available checkpoints
    checkpoint_dir = '/home/htiuser02/music/music/unconditional_generation/outputs/checkpoints'
    best_model_path = os.path.join(checkpoint_dir, 'best_model')

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
            print(f"Found best model checkpoint: {checkpoint_path}")
        else:
            # Check for any epoch checkpoint
            for epoch in [120, 110, 100, 1]:
                epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}')
                if os.path.exists(epoch_path):
                    checkpoint_path = epoch_path
                    print(f"Found checkpoint: {checkpoint_path}")
                    break

    result = test_generation(checkpoint_path, args.num_bars)
