# Unconditional Music Generation

GPT-2 based symbolic music generation using REMI representation.

## Project Structure

```
unconditional_generation/
├── src/
│   ├── dataset.py          # Dataset processing
│   ├── evaluate.py         # MusDr evaluation metrics
│   ├── generate.py         # MIDI generation script
│   ├── midi_processor.py   # MIDI ↔ REMI conversion
│   ├── midi_to_wav.py      # MIDI to WAV conversion
│   ├── model.py            # MusicGPT2 model
│   ├── train.py            # Training script
│   └── vocab.py            # Vocabulary handling
├── outputs/
│   ├── checkpoints/        # Model checkpoints (epoch 80, 100, 120)
│   ├── processed_data.pkl  # Preprocessed training data
│   └── training_config.json
├── evaluation/             # Evaluation results
├── final_outputs/
│   ├── midi/               # Generated MIDI files (20 samples)
│   └── wav/                # Generated WAV files (20 samples)
├── report.md               # Detailed project report
└── README.md
```

## Requirements

```bash
pip install torch transformers pretty_midi miditoolkit pyfluidsynth numpy pandas
```

## Usage

### Training

```bash
python src/train.py \
  --data_path /path/to/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi \
  --output_dir outputs \
  --epochs 120
```

### Generation

```bash
python src/generate.py \
  --checkpoint outputs/checkpoints/checkpoint_epoch_80 \
  --vocab_path /path/to/dictionary.pkl \
  --output_dir final_outputs/midi \
  --num_samples 20 \
  --num_bars 32 \
  --top_k 8 \
  --temperature 1.0
```

### Evaluation

```bash
python src/evaluate.py \
  --midi_dir final_outputs/midi \
  --output_dir evaluation
```

### MIDI to WAV Conversion

```bash
python src/midi_to_wav.py \
  --input_dir final_outputs/midi \
  --output_dir final_outputs/wav \
  --soundfont /path/to/soundfont.sf2
```

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Checkpoint | Epoch 80 |
| top_k | 8 |
| temperature | 1.0 |

## Evaluation Results

| Source | H1 | H4 | GS |
|--------|-----|-----|-----|
| Training Data | 2.070 | 2.528 | 0.909 |
| Generated (Best) | 2.291 | 2.701 | 0.923 |

- **H1**: 1-bar pitch class entropy
- **H4**: 4-bar pitch class entropy
- **GS**: Groove similarity

## Output

- 20 MIDI files (32 bars each)
- 20 WAV files

## References

- REMI: Pop Music Transformer (Huang & Yang, 2020)
- GPT-2 (Radford et al., 2019)
- Pop1K7 Dataset
