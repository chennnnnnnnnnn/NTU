# Continuation Generation

Music continuation generation using GPT-2 model from Task 1.

## Task

Given a prompt MIDI (8 bars), generate continuation for 24 bars (Total: 32 bars).

## Project Structure

```
Continuation_generation/
├── src/
│   ├── generate_all.py       # Main generation script
│   ├── verify_midi.py        # Verification script
│   ├── midi_processor.py     # MIDI-REMI conversion
│   └── vocab.py              # Vocabulary handling
├── prompt_song/              # Input prompt MIDI files (8 bars)
│   ├── song_1.mid
│   ├── song_2.mid
│   └── song_3.mid
├── final_outputs/
│   ├── midi/                 # Generated MIDI files (9 files)
│   └── wav/                  # Generated WAV files (9 files)
├── report.md                 # Detailed report
└── README.md
```

## Configurations

| Config | top_k | temperature |
|--------|-------|-------------|
| config1 | 8 | 1.0 |
| config2 | 10 | 1.15 |
| config3 | 5 | 0.9 |

## Usage

### Generate Continuations

```bash
python3 src/generate_all.py
```

### Verify Generated Files

```bash
python3 src/verify_midi.py
```

## Output

- 9 MIDI files: 3 prompts × 3 configurations
- 9 WAV files: Converted from MIDI

## Verification

All generated files are verified for:
1. Total 32 bars
2. First 8 bars match prompt (100%)
3. All bars have notes

## Model

Uses checkpoint from Task 1 (Unconditional Generation):
- Model: GPT-2 based transformer
- Checkpoint: Epoch 80
- Vocab: Pop1K7 REMI representation
