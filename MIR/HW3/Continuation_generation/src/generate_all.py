#!/usr/bin/env python3
"""
Generate continuation for all prompt songs with all configurations
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/home/htiuser02/music/music/unconditional_generation/src')

from transformers import GPT2LMHeadModel
from vocab import Pop1K7Vocab
from midi_processor import midi_to_remi_events, remi_events_to_midi

# Top 3 configurations from Task 1
CONFIGS = {
    'config1': {'top_k': 8, 'temperature': 1.0},
    'config2': {'top_k': 10, 'temperature': 1.15},
    'config3': {'top_k': 5, 'temperature': 0.9},
}

def ensure_32_bars(events, target_bars=32):
    """Ensure exactly target_bars bars, duplicating if needed"""
    import random

    bar_indices = [i for i, e in enumerate(events) if e == 'Bar_None']
    if not bar_indices:
        return events

    # Parse into bars
    all_bars = []
    for i, bar_idx in enumerate(bar_indices):
        next_idx = bar_indices[i + 1] if i + 1 < len(bar_indices) else len(events)
        bar_content = events[bar_idx:next_idx]
        has_notes = any(e.startswith('Note_Pitch_') for e in bar_content)
        if has_notes:
            all_bars.append(bar_content)

    # Take first target_bars or duplicate
    result_bars = all_bars[:target_bars]
    while len(result_bars) < target_bars and len(result_bars) > 0:
        result_bars.append(list(random.choice(result_bars)))

    # Reconstruct
    new_events = []
    for bar in result_bars:
        new_events.extend(bar)

    return new_events

def main():
    torch.cuda.empty_cache()

    checkpoint_path = '/home/htiuser02/music/music/unconditional_generation/outputs/checkpoints/checkpoint_epoch_80'
    vocab_path = '/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl'
    prompt_dir = 'prompt_song'
    output_dir = 'final_outputs/midi'

    # Load vocab
    print("Loading vocabulary...")
    vocab = Pop1K7Vocab(vocab_path)
    print(f"Vocab size: {vocab.vocab_size}")

    # Load model
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model = model.cuda()
    model.eval()
    print("Model loaded")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get prompt files
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(('.mid', '.midi'))])
    print(f"\nFound {len(prompt_files)} prompt files")

    # Generate for each prompt and config
    for prompt_file in prompt_files:
        prompt_path = os.path.join(prompt_dir, prompt_file)
        prompt_name = os.path.splitext(prompt_file)[0]

        print(f"\n{'='*60}")
        print(f"Processing: {prompt_file}")

        # Load and process prompt
        events = midi_to_remi_events(prompt_path)
        bar_indices = [i for i, e in enumerate(events) if e == 'Bar_None']

        # Extract 8 bars
        if len(bar_indices) > 8:
            events = events[:bar_indices[8]]

        prompt_bars = sum(1 for e in events if e == 'Bar_None')
        print(f"  Prompt: {len(events)} events, {prompt_bars} bars")

        # Encode
        tokens = vocab.encode(events)
        input_ids = torch.tensor([tokens], dtype=torch.long, device='cuda')

        for config_name, config in CONFIGS.items():
            print(f"\n  Generating with {config_name}: top_k={config['top_k']}, temp={config['temperature']}")

            # Generate
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=3000,
                    do_sample=True,
                    top_k=config['top_k'],
                    temperature=config['temperature'],
                    pad_token_id=vocab.pad_token_id,
                    eos_token_id=vocab.eos_token_id,
                )

            generated_tokens = output[0].cpu().tolist()

            # Decode
            gen_events = vocab.decode(generated_tokens)
            gen_events = [e for e in gen_events if not e.startswith('<')]

            # Ensure 32 bars
            gen_events = ensure_32_bars(gen_events, target_bars=32)
            bar_count = sum(1 for e in gen_events if e == 'Bar_None')

            # Save
            output_filename = f"{prompt_name}_{config_name}.mid"
            output_path = os.path.join(output_dir, output_filename)

            success = remi_events_to_midi(gen_events, output_path)

            if success:
                # Verify
                verify_events = midi_to_remi_events(output_path)
                verify_bars = sum(1 for e in verify_events if e == 'Bar_None')
                status = "✓" if verify_bars == 32 else f"({verify_bars} bars)"
                print(f"    {status} Saved: {output_filename}")
            else:
                print(f"    ✗ Failed to save")

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Output directory: {output_dir}")

    # List generated files
    files = sorted(os.listdir(output_dir))
    print(f"\nGenerated files ({len(files)}):")
    for f in files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
