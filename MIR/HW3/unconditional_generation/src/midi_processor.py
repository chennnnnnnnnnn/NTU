"""
MIDI to REMI Event Processor with Chord Detection
Processes MIDI files into REMI token sequences with chord information
"""

import os
import pickle
import numpy as np
from glob import glob
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
from tqdm import tqdm
import random


# Constants for REMI representation
TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR
POSITION_RESOLUTION = 16  # 16th note resolution per bar


def quantize_tempo(tempo: float) -> int:
    """Quantize tempo to nearest value in range [32, 224] with step 3"""
    tempo = int(round(tempo))
    tempo = max(32, min(224, tempo))
    # Round to nearest multiple of 3 starting from 32
    tempo = 32 + round((tempo - 32) / 3) * 3
    return min(tempo, 224)


def quantize_velocity(velocity: int) -> int:
    """Quantize velocity to range [40, 86] with step 2"""
    velocity = max(40, min(86, velocity))
    return 40 + round((velocity - 40) / 2) * 2


def quantize_duration(duration: int) -> int:
    """Quantize duration to predefined values"""
    durations = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 1920]
    # Find nearest duration
    duration = min(duration, 1920)
    nearest = min(durations, key=lambda x: abs(x - duration))
    return nearest


def tick_to_position(tick: int, ticks_per_bar: int = TICKS_PER_BAR) -> Tuple[int, int]:
    """Convert tick to (bar, position within bar)"""
    bar = tick // ticks_per_bar
    position_in_bar = tick % ticks_per_bar
    # Convert to 16th note position (0-15)
    position = int(position_in_bar / ticks_per_bar * POSITION_RESOLUTION)
    position = min(position, POSITION_RESOLUTION - 1)
    return bar, position


# Chord detection utilities
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates (intervals from root)
CHORD_TEMPLATES = {
    'M': [0, 4, 7],           # Major
    'm': [0, 3, 7],           # Minor
    '7': [0, 4, 7, 10],       # Dominant 7th
    'M7': [0, 4, 7, 11],      # Major 7th
    'm7': [0, 3, 7, 10],      # Minor 7th
    'o': [0, 3, 6],           # Diminished
    'o7': [0, 3, 6, 9],       # Diminished 7th
    '/o7': [0, 3, 6, 10],     # Half-diminished 7th
    '+': [0, 4, 8],           # Augmented
    'sus2': [0, 2, 7],        # Suspended 2nd
    'sus4': [0, 5, 7],        # Suspended 4th
}


def detect_chord(notes_in_segment: List[Note]) -> str:
    """Detect chord from notes in a time segment"""
    if not notes_in_segment:
        return 'Chord_N_N'

    # Get pitch classes present
    pitch_classes = set()
    for note in notes_in_segment:
        pitch_classes.add(note.pitch % 12)

    if len(pitch_classes) < 3:
        return 'Chord_N_N'

    # Try to match against chord templates
    best_match = None
    best_score = 0

    for root_idx in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            # Create expected pitch classes for this chord
            expected_pcs = set((root_idx + interval) % 12 for interval in intervals)

            # Calculate match score (Jaccard similarity)
            intersection = len(pitch_classes & expected_pcs)
            union = len(pitch_classes | expected_pcs)
            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_match = f'Chord_{PITCH_CLASSES[root_idx]}_{chord_type}'

    if best_score >= 0.6:  # Threshold for accepting a chord
        return best_match
    return 'Chord_N_N'


def midi_to_remi_events(midi_path: str, max_bars: int = 64) -> List[str]:
    """
    Convert MIDI file to REMI event sequence with chord detection

    Returns:
        List of event strings in REMI format
    """
    try:
        midi = miditoolkit.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return []

    # Get all notes from all tracks
    all_notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        return []

    # Sort notes by start time, then by pitch
    all_notes.sort(key=lambda x: (x.start, x.pitch))

    # Get tempo changes
    tempo_changes = midi.tempo_changes if midi.tempo_changes else []
    if not tempo_changes:
        tempo_changes = [miditoolkit.TempoChange(tempo=120, time=0)]

    # Determine number of bars based on note start times (not end times)
    # This prevents creating empty bars when notes extend past their bar
    max_start_tick = max(note.start for note in all_notes)
    total_bars = (max_start_tick // TICKS_PER_BAR) + 1
    total_bars = min(total_bars, max_bars)

    # Build event sequence
    events = []

    # Group notes by bar
    notes_by_bar = defaultdict(list)
    for note in all_notes:
        bar, _ = tick_to_position(note.start)
        if bar < total_bars:
            notes_by_bar[bar].append(note)

    # Build tempo lookup
    def get_tempo_at_tick(tick):
        tempo = 120
        for tc in tempo_changes:
            if tc.time <= tick:
                tempo = tc.tempo
            else:
                break
        return quantize_tempo(tempo)

    current_tempo = None

    for bar in range(total_bars):
        bar_start_tick = bar * TICKS_PER_BAR
        bar_end_tick = (bar + 1) * TICKS_PER_BAR

        # Add bar event
        events.append('Bar_None')

        # Add tempo event if changed
        tempo = get_tempo_at_tick(bar_start_tick)
        if tempo != current_tempo:
            events.append(f'Tempo_{tempo}')
            current_tempo = tempo

        # Detect chord for this bar
        bar_notes = notes_by_bar[bar]
        chord = detect_chord(bar_notes)
        events.append(chord)

        # Group notes by position within bar
        notes_by_position = defaultdict(list)
        for note in bar_notes:
            _, position = tick_to_position(note.start)
            notes_by_position[position].append(note)

        # Process notes at each position
        for position in sorted(notes_by_position.keys()):
            position_notes = notes_by_position[position]

            # Add position event
            events.append(f'Beat_{position}')

            # Add note events (sorted by pitch for consistency)
            for note in sorted(position_notes, key=lambda x: x.pitch):
                # Pitch (constrain to piano range)
                pitch = max(22, min(107, note.pitch))
                events.append(f'Note_Pitch_{pitch}')

                # Velocity
                velocity = quantize_velocity(note.velocity)
                events.append(f'Note_Velocity_{velocity}')

                # Duration
                duration = quantize_duration(note.end - note.start)
                events.append(f'Note_Duration_{duration}')

    return events


def remi_events_to_midi(events: List[str], output_path: str, ticks_per_beat: int = 480) -> bool:
    """
    Convert REMI event sequence back to MIDI file

    Returns:
        True if successful, False otherwise
    """
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    track = Instrument(program=0, is_drum=False, name='Piano')

    current_bar = 0  # Start at 0 instead of -1
    current_position = 0
    current_tempo = 120
    first_bar_seen = False

    notes = []
    tempo_changes = [miditoolkit.TempoChange(tempo=120, time=0)]  # Default tempo

    i = 0
    while i < len(events):
        event = events[i]

        if event == 'Bar_None':
            if first_bar_seen:
                current_bar += 1
            else:
                first_bar_seen = True
                current_bar = 0
            current_position = 0  # Reset position at each bar
        elif event.startswith('Beat_'):
            try:
                current_position = int(event.split('_')[1])
            except (ValueError, IndexError):
                pass
        elif event.startswith('Tempo_'):
            try:
                tempo = int(event.split('_')[1])
                if tempo != current_tempo:
                    current_tempo = tempo
                    tick = max(0, current_bar * TICKS_PER_BAR)  # Ensure non-negative
                    tempo_changes.append(miditoolkit.TempoChange(tempo=tempo, time=tick))
            except (ValueError, IndexError):
                pass
        elif event.startswith('Chord_'):
            pass  # Skip chord events for MIDI output
        elif event.startswith('Note_Pitch_'):
            try:
                pitch = int(event.split('_')[2])

                # Get velocity
                velocity = 64  # default
                if i + 1 < len(events) and events[i + 1].startswith('Note_Velocity_'):
                    velocity = int(events[i + 1].split('_')[2])
                    i += 1

                # Get duration
                duration = 480  # default
                if i + 1 < len(events) and events[i + 1].startswith('Note_Duration_'):
                    duration = int(events[i + 1].split('_')[2])
                    i += 1

                # Ensure duration is positive
                duration = max(120, duration)

                # Calculate start tick (ensure non-negative)
                tick_position = current_position * (TICKS_PER_BAR // POSITION_RESOLUTION)
                start_tick = max(0, current_bar * TICKS_PER_BAR + tick_position)
                end_tick = start_tick + duration

                notes.append(Note(
                    start=start_tick,
                    end=end_tick,
                    pitch=pitch,
                    velocity=velocity
                ))
            except (ValueError, IndexError):
                pass

        i += 1

    if not notes:
        return False

    # Sort notes by start time
    notes.sort(key=lambda x: (x.start, x.pitch))
    track.notes = notes
    midi.instruments.append(track)

    # Remove duplicate tempo changes and sort
    seen_times = set()
    unique_tempo_changes = []
    for tc in sorted(tempo_changes, key=lambda x: x.time):
        if tc.time not in seen_times:
            unique_tempo_changes.append(tc)
            seen_times.add(tc.time)
    midi.tempo_changes = unique_tempo_changes

    try:
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        midi.dump(output_path)
        return True
    except Exception as e:
        print(f"Error saving MIDI: {e}")
        return False


def process_dataset(
    midi_dir: str,
    output_dir: str,
    vocab,
    max_bars: int = 64,
    target_bars: int = 32,
    augment: bool = True
) -> Tuple[List[List[int]], List[str]]:
    """
    Process all MIDI files in directory to token sequences

    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Output directory for processed data
        vocab: Vocabulary object
        max_bars: Maximum bars to extract from each file
        target_bars: Target number of bars for training sequences
        augment: Whether to apply augmentation

    Returns:
        Tuple of (list of token sequences, list of source file paths)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(glob(os.path.join(midi_dir, '**', ext), recursive=True))

    print(f"Found {len(midi_files)} MIDI files")

    all_sequences = []
    all_sources = []

    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        events = midi_to_remi_events(midi_path, max_bars=max_bars)

        if len(events) < 100:  # Skip very short pieces
            continue

        # Convert events to tokens
        tokens = vocab.encode(events)

        # Extract 32-bar segments
        segments = extract_bar_segments(tokens, vocab, target_bars=target_bars)

        for seg_tokens in segments:
            all_sequences.append(seg_tokens)
            all_sources.append(midi_path)

            # Data augmentation
            if augment:
                # Pitch transposition (-5 to +6 semitones)
                for transpose in [-3, -1, 1, 3]:
                    aug_tokens = transpose_sequence(seg_tokens, vocab, transpose)
                    if aug_tokens:
                        all_sequences.append(aug_tokens)
                        all_sources.append(f"{midi_path}__transpose_{transpose}")

    print(f"Total sequences: {len(all_sequences)}")
    return all_sequences, all_sources


def extract_bar_segments(
    tokens: List[int],
    vocab,
    target_bars: int = 32,
    overlap_bars: int = 8
) -> List[List[int]]:
    """
    Extract fixed-length bar segments from token sequence

    Args:
        tokens: Full token sequence
        vocab: Vocabulary object
        target_bars: Target number of bars per segment
        overlap_bars: Overlap between consecutive segments

    Returns:
        List of token segments
    """
    # Find bar token positions
    bar_token = vocab.bar_token
    bar_positions = [i for i, t in enumerate(tokens) if t == bar_token]

    if len(bar_positions) < target_bars:
        return []

    segments = []
    step = target_bars - overlap_bars

    for start_bar_idx in range(0, len(bar_positions) - target_bars + 1, step):
        start_pos = bar_positions[start_bar_idx]
        end_bar_idx = start_bar_idx + target_bars

        if end_bar_idx < len(bar_positions):
            end_pos = bar_positions[end_bar_idx]
        else:
            end_pos = len(tokens)

        segment = tokens[start_pos:end_pos]
        segments.append(segment)

    return segments


def transpose_sequence(tokens: List[int], vocab, semitones: int) -> Optional[List[int]]:
    """
    Transpose pitch tokens by given semitones

    Args:
        tokens: Token sequence
        vocab: Vocabulary object
        semitones: Number of semitones to transpose

    Returns:
        Transposed token sequence or None if transposition would go out of range
    """
    result = []

    for token in tokens:
        event = vocab.idx2event.get(token, '')

        if event.startswith('Note_Pitch_'):
            try:
                pitch = int(event.split('_')[2])
                new_pitch = pitch + semitones

                # Check bounds (piano range: 22-107)
                if new_pitch < 22 or new_pitch > 107:
                    return None

                new_event = f'Note_Pitch_{new_pitch}'
                new_token = vocab.event2idx.get(new_event)

                if new_token is None:
                    return None

                result.append(new_token)
            except (ValueError, IndexError):
                result.append(token)
        elif event.startswith('Chord_') and event != 'Chord_N_N':
            # Transpose chord root
            try:
                parts = event.split('_')
                root = parts[1]
                chord_type = parts[2]

                root_idx = PITCH_CLASSES.index(root)
                new_root_idx = (root_idx + semitones) % 12
                new_root = PITCH_CLASSES[new_root_idx]

                new_event = f'Chord_{new_root}_{chord_type}'
                new_token = vocab.event2idx.get(new_event)

                if new_token is not None:
                    result.append(new_token)
                else:
                    result.append(token)
            except (ValueError, IndexError):
                result.append(token)
        else:
            result.append(token)

    return result


if __name__ == '__main__':
    # Test MIDI processing
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from vocab import Pop1K7Vocab

    # Load vocabulary
    vocab_path = '/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl'
    vocab = Pop1K7Vocab(vocab_path)

    # Test with a sample MIDI file
    test_midi_dir = '/home/htiuser02/music/music/Pop1K7/Pop1K7/midi_analyzed/src_001/'
    midi_files = glob(os.path.join(test_midi_dir, '*.mid'))

    if midi_files:
        test_file = midi_files[0]
        print(f"Testing with: {test_file}")

        events = midi_to_remi_events(test_file, max_bars=32)
        print(f"Number of events: {len(events)}")
        print(f"First 20 events: {events[:20]}")

        # Test reconstruction
        output_path = '/tmp/test_reconstruction.mid'
        success = remi_events_to_midi(events, output_path)
        print(f"Reconstruction successful: {success}")
