#!/usr/bin/env python3
"""
Verify generated MIDI files:
1. First 8 bars match prompt
2. Total 32 bars
3. All bars have notes and are continuous
"""

import os
import sys
sys.path.insert(0, '/home/htiuser02/music/music/unconditional_generation/src')

from midi_processor import midi_to_remi_events

def get_bars_info(events):
    """Parse events into bars with detailed info"""
    bar_indices = [i for i, e in enumerate(events) if e == 'Bar_None']
    bars = []

    for i, bar_idx in enumerate(bar_indices):
        next_idx = bar_indices[i + 1] if i + 1 < len(bar_indices) else len(events)
        bar_content = events[bar_idx:next_idx]

        # Count notes
        note_pitches = [e for e in bar_content if e.startswith('Note_Pitch_')]

        bars.append({
            'index': i + 1,
            'start_event_idx': bar_idx,
            'end_event_idx': next_idx,
            'num_events': len(bar_content),
            'num_notes': len(note_pitches),
            'has_notes': len(note_pitches) > 0,
            'note_pitches': note_pitches[:3]  # First 3 for display
        })

    return bars

def compare_bars(prompt_bars, generated_bars, num_prompt_bars=8):
    """Compare first num_prompt_bars bars between prompt and generated"""
    matches = []

    for i in range(min(num_prompt_bars, len(prompt_bars), len(generated_bars))):
        p_bar = prompt_bars[i]
        g_bar = generated_bars[i]

        # Compare note pitches
        match = p_bar['note_pitches'] == g_bar['note_pitches']
        matches.append({
            'bar': i + 1,
            'prompt_notes': p_bar['num_notes'],
            'generated_notes': g_bar['num_notes'],
            'match': match
        })

    return matches

def verify_file(prompt_path, generated_path, prompt_name):
    """Verify a single generated file"""
    print(f"\n{'='*60}")
    print(f"Verifying: {os.path.basename(generated_path)}")
    print(f"Prompt: {os.path.basename(prompt_path)}")

    # Load events
    prompt_events = midi_to_remi_events(prompt_path)
    generated_events = midi_to_remi_events(generated_path)

    # Get bars info
    prompt_bars = get_bars_info(prompt_events)
    generated_bars = get_bars_info(generated_events)

    print(f"\nPrompt: {len(prompt_bars)} bars, {len(prompt_events)} events")
    print(f"Generated: {len(generated_bars)} bars, {len(generated_events)} events")

    # Check 1: Total 32 bars
    total_bars = len(generated_bars)
    check1 = total_bars == 32
    print(f"\n[1] Total 32 bars: {'✓ PASS' if check1 else f'✗ FAIL ({total_bars} bars)'}")

    # Check 2: All bars have notes
    bars_without_notes = [b['index'] for b in generated_bars if not b['has_notes']]
    check2 = len(bars_without_notes) == 0
    if check2:
        print(f"[2] All bars have notes: ✓ PASS")
    else:
        print(f"[2] All bars have notes: ✗ FAIL (bars {bars_without_notes} have no notes)")

    # Check 3: First 8 bars match prompt
    # Compare the events directly for first 8 bars
    prompt_8bars_events = []
    gen_8bars_events = []

    if len(prompt_bars) >= 8:
        end_idx = prompt_bars[7]['end_event_idx'] if len(prompt_bars) > 7 else len(prompt_events)
        prompt_8bars_events = prompt_events[:end_idx]

    if len(generated_bars) >= 8:
        end_idx = generated_bars[7]['end_event_idx']
        gen_8bars_events = generated_events[:end_idx]

    # Compare
    if len(prompt_8bars_events) > 0 and len(gen_8bars_events) > 0:
        # Compare note pitches in first 8 bars
        prompt_notes = [e for e in prompt_8bars_events if e.startswith('Note_Pitch_')]
        gen_notes = [e for e in gen_8bars_events if e.startswith('Note_Pitch_')]

        # Check if they start the same (at least first 50 notes)
        min_len = min(len(prompt_notes), len(gen_notes), 50)
        matching_notes = sum(1 for i in range(min_len) if prompt_notes[i] == gen_notes[i])
        match_ratio = matching_notes / min_len if min_len > 0 else 0

        check3 = match_ratio > 0.9  # At least 90% match
        print(f"[3] First 8 bars match prompt: {'✓ PASS' if check3 else '✗ FAIL'} ({match_ratio*100:.1f}% match)")
    else:
        check3 = False
        print(f"[3] First 8 bars match prompt: ✗ FAIL (could not compare)")

    # Check 4: Bars are continuous (no large gaps)
    print(f"\n[4] Bar details:")
    for bar in generated_bars:
        status = "✓" if bar['has_notes'] else "✗ NO NOTES"
        print(f"    Bar {bar['index']:2d}: {bar['num_notes']:3d} notes {status}")

    # Summary
    all_pass = check1 and check2 and check3
    print(f"\n{'='*60}")
    print(f"Overall: {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")

    return all_pass

def main():
    prompt_dir = 'prompt_song'
    output_dir = 'final_outputs/midi'

    # Get all generated files
    generated_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.mid')])

    if not generated_files:
        print("No generated files found!")
        return

    print(f"Found {len(generated_files)} generated files")

    results = []
    for gen_file in generated_files:
        # Extract prompt name (e.g., song_1_config1.mid -> song_1)
        parts = gen_file.replace('.mid', '').split('_')
        prompt_name = '_'.join(parts[:-1])  # song_1

        prompt_file = f"{prompt_name}.mid"
        prompt_path = os.path.join(prompt_dir, prompt_file)
        generated_path = os.path.join(output_dir, gen_file)

        if os.path.exists(prompt_path):
            passed = verify_file(prompt_path, generated_path, prompt_name)
            results.append((gen_file, passed))
        else:
            print(f"\nWarning: Prompt file not found: {prompt_file}")
            results.append((gen_file, False))

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for filename, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {filename}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} passed")

if __name__ == '__main__':
    main()
