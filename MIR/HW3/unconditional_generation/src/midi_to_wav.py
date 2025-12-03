"""
MIDI to WAV Conversion Script
Uses FluidSynth or pretty_midi for synthesis
"""

import os
import sys
import argparse
from glob import glob
from typing import Optional
import numpy as np
from tqdm import tqdm

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def midi_to_wav_pretty_midi(
    midi_path: str,
    wav_path: str,
    sample_rate: int = 44100
) -> bool:
    """
    Convert MIDI to WAV using pretty_midi's built-in synthesizer

    Args:
        midi_path: Input MIDI file path
        wav_path: Output WAV file path
        sample_rate: Audio sample rate

    Returns:
        True if successful
    """
    if not PRETTY_MIDI_AVAILABLE or not SCIPY_AVAILABLE:
        print("Error: pretty_midi and scipy are required for this function")
        return False

    try:
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Synthesize audio
        audio = midi_data.fluidsynth(fs=sample_rate)

        # Normalize audio
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save as WAV
        wavfile.write(wav_path, sample_rate, audio_int16)

        return True

    except Exception as e:
        print(f"Error converting {midi_path}: {e}")
        return False


def midi_to_wav_fluidsynth(
    midi_path: str,
    wav_path: str,
    soundfont_path: str,
    sample_rate: int = 44100
) -> bool:
    """
    Convert MIDI to WAV using FluidSynth command-line tool

    Args:
        midi_path: Input MIDI file path
        wav_path: Output WAV file path
        soundfont_path: Path to SoundFont file (.sf2)
        sample_rate: Audio sample rate

    Returns:
        True if successful
    """
    import subprocess

    try:
        cmd = [
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_path,
            '-F', wav_path,
            '-r', str(sample_rate)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FluidSynth error: {result.stderr}")
            return False

        return os.path.exists(wav_path)

    except FileNotFoundError:
        print("FluidSynth not found. Please install it: sudo apt-get install fluidsynth")
        return False
    except Exception as e:
        print(f"Error converting {midi_path}: {e}")
        return False


def simple_midi_to_audio(
    midi_path: str,
    wav_path: str,
    sample_rate: int = 44100
) -> bool:
    """
    Simple MIDI to audio conversion using sine wave synthesis

    This is a fallback when FluidSynth is not available
    """
    try:
        import mido
        from scipy.io import wavfile

        mid = mido.MidiFile(midi_path)

        # Get all note events
        notes = []
        current_time = 0

        for msg in mid:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append({
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'start': current_time
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Find the note and set end time
                for note in reversed(notes):
                    if note['pitch'] == msg.note and 'end' not in note:
                        note['end'] = current_time
                        break

        # Set default end for notes without note_off
        duration = current_time + 1.0
        for note in notes:
            if 'end' not in note:
                note['end'] = note['start'] + 0.5

        # Synthesize audio
        num_samples = int(duration * sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)

        for note in notes:
            freq = 440.0 * (2.0 ** ((note['pitch'] - 69) / 12.0))
            start_sample = int(note['start'] * sample_rate)
            end_sample = int(note['end'] * sample_rate)
            end_sample = min(end_sample, num_samples)

            if start_sample >= end_sample:
                continue

            t = np.arange(end_sample - start_sample) / sample_rate
            amplitude = note['velocity'] / 127.0 * 0.3

            # Simple ADSR envelope
            attack = 0.01
            decay = 0.1
            sustain = 0.7
            release = 0.1

            envelope = np.ones_like(t)
            attack_samples = int(attack * sample_rate)
            decay_samples = int(decay * sample_rate)
            release_samples = int(release * sample_rate)

            if attack_samples > 0:
                envelope[:min(attack_samples, len(t))] = np.linspace(0, 1, min(attack_samples, len(t)))

            if release_samples > 0 and len(t) > release_samples:
                envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)

            # Generate sine wave
            wave = amplitude * envelope * np.sin(2 * np.pi * freq * t)
            audio[start_sample:end_sample] += wave

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save as WAV
        wavfile.write(wav_path, sample_rate, audio_int16)

        return True

    except Exception as e:
        print(f"Error in simple synthesis for {midi_path}: {e}")
        return False


def convert_directory(
    midi_dir: str,
    wav_dir: str,
    soundfont_path: Optional[str] = None,
    sample_rate: int = 44100,
    use_fluidsynth: bool = True
) -> int:
    """
    Convert all MIDI files in a directory to WAV

    Args:
        midi_dir: Input directory with MIDI files
        wav_dir: Output directory for WAV files
        soundfont_path: Path to SoundFont file (for FluidSynth)
        sample_rate: Audio sample rate
        use_fluidsynth: Whether to use FluidSynth (falls back to simple synthesis)

    Returns:
        Number of successfully converted files
    """
    os.makedirs(wav_dir, exist_ok=True)

    # Find MIDI files
    midi_files = glob(os.path.join(midi_dir, '*.mid'))
    midi_files.extend(glob(os.path.join(midi_dir, '*.midi')))

    print(f"Found {len(midi_files)} MIDI files to convert")

    success_count = 0

    for midi_path in tqdm(midi_files, desc="Converting to WAV"):
        base_name = os.path.splitext(os.path.basename(midi_path))[0]
        wav_path = os.path.join(wav_dir, f'{base_name}.wav')

        success = False

        # Try FluidSynth first if available
        if use_fluidsynth and soundfont_path:
            success = midi_to_wav_fluidsynth(
                midi_path, wav_path, soundfont_path, sample_rate
            )

        # Fall back to pretty_midi
        if not success and PRETTY_MIDI_AVAILABLE:
            success = midi_to_wav_pretty_midi(midi_path, wav_path, sample_rate)

        # Fall back to simple synthesis
        if not success:
            success = simple_midi_to_audio(midi_path, wav_path, sample_rate)

        if success:
            success_count += 1

    print(f"Successfully converted {success_count}/{len(midi_files)} files")
    return success_count


def main():
    parser = argparse.ArgumentParser(description='Convert MIDI files to WAV')

    parser.add_argument('--midi_dir', type=str, required=True,
                        help='Directory containing MIDI files')
    parser.add_argument('--wav_dir', type=str, required=True,
                        help='Output directory for WAV files')
    parser.add_argument('--soundfont', type=str, default=None,
                        help='Path to SoundFont file (.sf2)')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--no_fluidsynth', action='store_true',
                        help='Do not use FluidSynth')

    args = parser.parse_args()

    # Check for common SoundFont locations
    soundfont_paths = [
        args.soundfont,
        '/usr/share/sounds/sf2/FluidR3_GM.sf2',
        '/usr/share/soundfonts/FluidR3_GM.sf2',
        '/usr/share/sounds/sf2/default.sf2',
        os.path.expanduser('~/.fluidsynth/default.sf2')
    ]

    soundfont_path = None
    for path in soundfont_paths:
        if path and os.path.exists(path):
            soundfont_path = path
            print(f"Using SoundFont: {soundfont_path}")
            break

    if not soundfont_path and not args.no_fluidsynth:
        print("Warning: No SoundFont found. Using simple synthesis.")
        print("For better quality, install a SoundFont:")
        print("  sudo apt-get install fluid-soundfont-gm")

    convert_directory(
        args.midi_dir,
        args.wav_dir,
        soundfont_path=soundfont_path,
        sample_rate=args.sample_rate,
        use_fluidsynth=not args.no_fluidsynth
    )


if __name__ == '__main__':
    main()
