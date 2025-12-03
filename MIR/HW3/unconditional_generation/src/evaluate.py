"""
Evaluation Script using MusDr Metrics
Evaluate generated music: H1, H4, GS, SI_short, SI_mid, SI_long
"""

import os
import sys
import json
import argparse
from glob import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import tempfile

# Add MusDr to path
MUSDR_PATH = '/home/htiuser02/music/music/MusDr'
sys.path.insert(0, MUSDR_PATH)

from musdr.eval_metrics import (
    compute_piece_pitch_entropy,
    compute_piece_groove_similarity,
    compute_structure_indicator
)
from musdr.side_utils import get_event_seq

# Add source directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import Pop1K7Vocab
from midi_processor import midi_to_remi_events


# Event encoding mappings for MusDr (adapted for our vocabulary)
def get_vocab_mappings(vocab: Pop1K7Vocab) -> Dict:
    """Get vocabulary mappings for MusDr evaluation"""

    # Find bar token
    bar_token = vocab.event2idx.get('Bar_None', None)

    # Find position/beat tokens
    pos_tokens = [vocab.event2idx[f'Beat_{i}'] for i in range(16)
                  if f'Beat_{i}' in vocab.event2idx]

    # Find pitch tokens
    pitch_tokens = [vocab.event2idx[f'Note_Pitch_{p}']
                    for p in range(22, 108)
                    if f'Note_Pitch_{p}' in vocab.event2idx]

    return {
        'bar_token': bar_token,
        'pos_tokens': pos_tokens,
        'pitch_tokens': pitch_tokens
    }


def tokens_to_event_seq(
    tokens: List[int],
    vocab: Pop1K7Vocab
) -> List[int]:
    """
    Convert our token sequence to MusDr event sequence format

    The key mapping needed:
    - Bar events
    - Position events
    - Pitch events (Note-On equivalent)
    """
    # Decode tokens to events
    events = vocab.decode(tokens)

    # Convert to a format compatible with MusDr
    # MusDr expects: Bar=192, Position=193-256, Note-On=0-127, Chords=322-392
    event_seq = []

    for event in events:
        if event == 'Bar_None':
            event_seq.append(192)  # MusDr Bar event
        elif event.startswith('Beat_'):
            try:
                pos = int(event.split('_')[1])
                # Map 0-15 to 193-256 range (position events)
                event_seq.append(193 + pos * 4)  # Scale to 64 positions
            except (ValueError, IndexError):
                pass
        elif event.startswith('Note_Pitch_'):
            try:
                pitch = int(event.split('_')[2])
                event_seq.append(pitch)  # Pitch as Note-On
            except (ValueError, IndexError):
                pass
        elif event.startswith('Chord_') and event != 'Chord_N_N':
            # Map chord to MusDr chord event range
            # Chord-Tone: 322-333, Chord-Type: 346-392
            try:
                parts = event.split('_')
                root = parts[1]
                chord_type = parts[2]

                pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                                 'F#', 'G', 'G#', 'A', 'A#', 'B']
                root_idx = pitch_classes.index(root)
                event_seq.append(322 + root_idx)  # Chord-Tone

                # Map chord type
                chord_types = ['M', 'm', '7', 'M7', 'm7', 'o', 'o7',
                               '/o7', '+', 'sus2', 'sus4']
                if chord_type in chord_types:
                    type_idx = chord_types.index(chord_type)
                    event_seq.append(346 + type_idx)  # Chord-Type
            except (ValueError, IndexError):
                pass

    return event_seq


def evaluate_single_piece(
    midi_path: str,
    vocab: Pop1K7Vocab,
    scapeplot_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a single piece using MusDr metrics

    Args:
        midi_path: Path to MIDI file
        vocab: Vocabulary object
        scapeplot_dir: Optional directory containing pre-computed scape plots

    Returns:
        Dictionary of metric values
    """
    # Convert MIDI to REMI events
    events = midi_to_remi_events(midi_path, max_bars=64)

    if len(events) < 50:
        return None

    # Encode to tokens
    tokens = vocab.encode(events)

    # Convert to MusDr format
    event_seq = tokens_to_event_seq(tokens, vocab)

    if len(event_seq) < 50:
        return None

    results = {}

    try:
        # H1: 1-bar pitch-class histogram entropy
        results['H1'] = compute_piece_pitch_entropy(
            event_seq, window_size=1,
            bar_ev_id=192, pitch_evs=range(128)
        )
    except Exception as e:
        results['H1'] = np.nan

    try:
        # H4: 4-bar pitch-class histogram entropy
        results['H4'] = compute_piece_pitch_entropy(
            event_seq, window_size=4,
            bar_ev_id=192, pitch_evs=range(128)
        )
    except Exception as e:
        results['H4'] = np.nan

    try:
        # GS: Grooving pattern similarity
        results['GS'] = compute_piece_groove_similarity(
            event_seq,
            bar_ev_id=192,
            pos_evs=range(193, 257),
            pitch_evs=range(128)
        )
    except Exception as e:
        results['GS'] = np.nan

    # Structure Indicators (require scape plots from audio)
    # These would require audio rendering and scape plot computation
    # For now, set to NaN if scape plots not available
    results['SI_short'] = np.nan
    results['SI_mid'] = np.nan
    results['SI_long'] = np.nan

    if scapeplot_dir:
        # Look for corresponding scape plot
        base_name = os.path.splitext(os.path.basename(midi_path))[0]
        scapeplot_path = os.path.join(scapeplot_dir, f'{base_name}.npy')

        if os.path.exists(scapeplot_path):
            try:
                results['SI_short'] = compute_structure_indicator(
                    scapeplot_path, low_bound_sec=3, upp_bound_sec=8
                )
                results['SI_mid'] = compute_structure_indicator(
                    scapeplot_path, low_bound_sec=8, upp_bound_sec=15
                )
                results['SI_long'] = compute_structure_indicator(
                    scapeplot_path, low_bound_sec=15, upp_bound_sec=128
                )
            except Exception as e:
                pass

    return results


def evaluate_directory(
    midi_dir: str,
    vocab: Pop1K7Vocab,
    scapeplot_dir: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluate all MIDI files in a directory

    Args:
        midi_dir: Directory containing MIDI files
        vocab: Vocabulary object
        scapeplot_dir: Optional scape plot directory
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with evaluation results
    """
    midi_files = glob(os.path.join(midi_dir, '*.mid'))
    midi_files.extend(glob(os.path.join(midi_dir, '*.midi')))

    print(f"Found {len(midi_files)} MIDI files to evaluate")

    results = []

    for midi_path in tqdm(midi_files, desc="Evaluating"):
        metrics = evaluate_single_piece(midi_path, vocab, scapeplot_dir)

        if metrics is not None:
            metrics['file'] = os.path.basename(midi_path)
            results.append(metrics)

    df = pd.DataFrame(results)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df


def compute_average_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute average metrics from evaluation results"""
    metrics = ['H1', 'H4', 'GS', 'SI_short', 'SI_mid', 'SI_long']
    averages = {}

    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                averages[metric] = values.mean()
                averages[f'{metric}_std'] = values.std()
            else:
                averages[metric] = np.nan
                averages[f'{metric}_std'] = np.nan

    return averages


def evaluate_all_configurations(
    generated_dir: str,
    checkpoint_epochs: List[int],
    vocab: Pop1K7Vocab,
    output_dir: str,
    training_output_dir: str = None
) -> pd.DataFrame:
    """
    Evaluate all checkpoint and sampling configuration combinations

    Args:
        generated_dir: Directory containing generated MIDI files
        checkpoint_epochs: List of checkpoint epochs
        vocab: Vocabulary object
        output_dir: Output directory for results
        training_output_dir: Directory containing training outputs (for loss values)

    Returns:
        DataFrame with all results
    """
    # Sampling configurations mapping
    SAMPLING_CONFIGS = {
        'config1': {'top_k': 5, 'temperature': 0.9},
        'config2': {'top_k': 8, 'temperature': 1.0},
        'config3': {'top_k': 10, 'temperature': 1.15},
    }

    configs = ['config1', 'config2', 'config3']
    all_results = []

    # Load training history for loss values
    loss_by_epoch = {}
    if training_output_dir is None:
        training_output_dir = '/home/htiuser02/music/music/unconditional_generation/outputs'

    history_path = os.path.join(training_output_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            # Loss is indexed from 0, epoch 1 = index 0
            for i, loss in enumerate(history.get('val_loss', [])):
                loss_by_epoch[i + 1] = loss

    for epoch in checkpoint_epochs:
        for config in configs:
            # Path to generated files for this combination
            config_dir = os.path.join(
                generated_dir, f'epoch_{epoch}', config
            )

            if not os.path.exists(config_dir):
                print(f"Skipping {config_dir} - not found")
                continue

            print(f"\nEvaluating epoch={epoch}, config={config}")

            # Evaluate
            df = evaluate_directory(config_dir, vocab)

            if len(df) > 0:
                # Compute averages
                avg_metrics = compute_average_metrics(df)
                avg_metrics['epoch'] = epoch
                avg_metrics['config'] = config
                avg_metrics['num_samples'] = len(df)

                # Add sampling config details
                avg_metrics['top_k'] = SAMPLING_CONFIGS[config]['top_k']
                avg_metrics['temperature'] = SAMPLING_CONFIGS[config]['temperature']

                # Add loss from training history
                avg_metrics['loss'] = loss_by_epoch.get(epoch, np.nan)

                all_results.append(avg_metrics)

                # Print summary
                print(f"  H1: {avg_metrics.get('H1', np.nan):.4f}")
                print(f"  H4: {avg_metrics.get('H4', np.nan):.4f}")
                print(f"  GS: {avg_metrics.get('GS', np.nan):.4f}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Generate the 10x12 formatted CSV as requested
    generate_formatted_csv(all_results, output_dir)

    return summary_df


def generate_formatted_csv(results: List[Dict], output_dir: str):
    """
    Generate the 10x12 CSV format as requested:
    Columns: Model, representation, event, loss, top-k, temperature, H1, H4, GS, SI_short, SI_mid, SI_long
    Rows: 9 configurations (3 epochs × 3 sampling configs)
    """
    rows = []

    for result in results:
        epoch = result.get('epoch', 0)
        config = result.get('config', '')

        row = {
            'Model': f'GPT2_epoch{epoch}',
            'representation': 'REMI',
            'event': 'chord',
            'loss': result.get('loss', np.nan),
            'top-k': result.get('top_k', 0),
            'temperature': result.get('temperature', 0),
            'H1': result.get('H1', np.nan),
            'H4': result.get('H4', np.nan),
            'GS': result.get('GS', np.nan),
            'SI_short': result.get('SI_short', np.nan),
            'SI_mid': result.get('SI_mid', np.nan),
            'SI_long': result.get('SI_long', np.nan),
        }
        rows.append(row)

    # Create DataFrame with specific column order
    columns = ['Model', 'representation', 'event', 'loss', 'top-k', 'temperature',
               'H1', 'H4', 'GS', 'SI_short', 'SI_mid', 'SI_long']

    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'evaluation_results_formatted.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nFormatted 10x12 CSV saved to {csv_path}")

    # Print the table
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS (10x12 Format)")
    print("=" * 100)
    print(df.to_string(index=False))

    return df


def find_best_configuration(summary_df: pd.DataFrame) -> Dict:
    """
    Find the best configuration based on metrics

    Uses a composite score: higher GS and balanced H1/H4 are preferred
    """
    if len(summary_df) == 0:
        return None

    # Create composite score
    # Higher GS is better (rhythmic consistency)
    # Moderate H1/H4 is preferred (not too random, not too repetitive)
    # Target H values around 2.5-3.5 (typical for good music)

    def score_row(row):
        gs_score = row.get('GS', 0) * 100  # Scale up
        h1_score = 10 - abs(row.get('H1', 3) - 3.0) * 2
        h4_score = 10 - abs(row.get('H4', 3) - 3.0) * 2
        return gs_score + h1_score + h4_score

    summary_df['score'] = summary_df.apply(score_row, axis=1)

    # Find best
    best_idx = summary_df['score'].idxmax()
    best_row = summary_df.loc[best_idx]

    return {
        'epoch': int(best_row['epoch']),
        'config': best_row['config'],
        'score': best_row['score'],
        'H1': best_row.get('H1', np.nan),
        'H4': best_row.get('H4', np.nan),
        'GS': best_row.get('GS', np.nan)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Generated Music')

    parser.add_argument('--midi_dir', type=str, required=True,
                        help='Directory containing MIDI files to evaluate')
    parser.add_argument('--vocab_path', type=str,
                        default='/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl',
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str,
                        default='/home/htiuser02/music/music/unconditional_generation/evaluation/',
                        help='Output directory for results')
    parser.add_argument('--scapeplot_dir', type=str, default=None,
                        help='Directory containing scape plots (optional)')
    parser.add_argument('--eval_all_configs', action='store_true',
                        help='Evaluate all checkpoint/config combinations')
    parser.add_argument('--checkpoint_epochs', type=int, nargs='+',
                        default=[100, 110, 120],
                        help='Checkpoint epochs to evaluate')

    args = parser.parse_args()

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Pop1K7Vocab(args.vocab_path)

    if args.eval_all_configs:
        # Evaluate all configurations
        summary_df = evaluate_all_configurations(
            args.midi_dir,
            args.checkpoint_epochs,
            vocab,
            args.output_dir
        )

        # Find best configuration
        best = find_best_configuration(summary_df)
        if best:
            print("\n" + "=" * 50)
            print("Best Configuration:")
            print(f"  Epoch: {best['epoch']}")
            print(f"  Config: {best['config']}")
            print(f"  Score: {best['score']:.2f}")
            print(f"  H1: {best['H1']:.4f}")
            print(f"  H4: {best['H4']:.4f}")
            print(f"  GS: {best['GS']:.4f}")

            # Save best config
            with open(os.path.join(args.output_dir, 'best_config.json'), 'w') as f:
                json.dump(best, f, indent=2)
    else:
        # Evaluate single directory
        output_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        df = evaluate_directory(
            args.midi_dir, vocab,
            scapeplot_dir=args.scapeplot_dir,
            output_path=output_path
        )

        # Print summary
        avg = compute_average_metrics(df)
        print("\n" + "=" * 50)
        print("Average Metrics:")
        for metric, value in avg.items():
            if not metric.endswith('_std'):
                print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
