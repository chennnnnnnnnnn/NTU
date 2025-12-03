"""
REMI Vocabulary with Chord for Pop Music Generation
Event types: Bar, Beat, Chord, Tempo, Note_Pitch, Note_Velocity, Note_Duration
"""

import json
import pickle
from typing import Dict, List, Tuple

# Special tokens
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# Event type definitions
BAR_EVENTS = ['Bar_None']
BEAT_EVENTS = [f'Beat_{i}' for i in range(16)]  # 16 positions per bar (16th note resolution)

# Chord events: 12 roots x 11 chord types + N_N (no chord)
CHORD_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_TYPES = ['M', 'm', '7', 'M7', 'm7', 'o', 'o7', '/o7', '+', 'sus2', 'sus4']
CHORD_EVENTS = [f'Chord_{root}_{ctype}' for root in CHORD_ROOTS for ctype in CHORD_TYPES] + ['Chord_N_N']

# Tempo events (range: 32-224 BPM with step 3)
TEMPO_EVENTS = [f'Tempo_{t}' for t in range(32, 225, 3)]

# Note pitch events (MIDI pitch 22-107 for piano range)
NOTE_PITCH_EVENTS = [f'Note_Pitch_{p}' for p in range(22, 108)]

# Note velocity events (40-86 with step 2)
NOTE_VELOCITY_EVENTS = [f'Note_Velocity_{v}' for v in range(40, 88, 2)]

# Note duration events (in ticks, 480 ticks per beat)
# 0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 1920
DURATION_VALUES = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800, 1920]
NOTE_DURATION_EVENTS = [f'Note_Duration_{d}' for d in DURATION_VALUES]


class REMIVocab:
    """REMI vocabulary with chord events"""

    def __init__(self):
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # Build vocabulary
        self.event2idx: Dict[str, int] = {}
        self.idx2event: Dict[int, str] = {}

        self._build_vocab()

    def _build_vocab(self):
        idx = 0

        # Special tokens first
        for token in self.special_tokens:
            self.event2idx[token] = idx
            self.idx2event[idx] = token
            idx += 1

        # Bar events
        for event in BAR_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Beat events
        for event in BEAT_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Chord events
        for event in CHORD_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Tempo events
        for event in TEMPO_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Note pitch events
        for event in NOTE_PITCH_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Note velocity events
        for event in NOTE_VELOCITY_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        # Note duration events
        for event in NOTE_DURATION_EVENTS:
            self.event2idx[event] = idx
            self.idx2event[idx] = event
            idx += 1

        self.vocab_size = idx

        # Store token type ranges for convenience
        self.bar_token = self.event2idx['Bar_None']
        self.beat_tokens = list(range(self.event2idx['Beat_0'], self.event2idx['Beat_0'] + 16))
        self.chord_token_start = self.event2idx['Chord_C_M']
        self.chord_token_end = self.event2idx['Chord_N_N'] + 1
        self.tempo_token_start = self.event2idx['Tempo_32']
        self.tempo_token_end = self.event2idx[f'Tempo_{224 - (224-32) % 3}'] + 1
        self.pitch_token_start = self.event2idx['Note_Pitch_22']
        self.pitch_token_end = self.event2idx['Note_Pitch_107'] + 1
        self.velocity_token_start = self.event2idx['Note_Velocity_40']
        self.velocity_token_end = self.event2idx['Note_Velocity_86'] + 1
        self.duration_token_start = self.event2idx['Note_Duration_0']
        self.duration_token_end = self.event2idx['Note_Duration_1920'] + 1

    @property
    def pad_token_id(self) -> int:
        return self.event2idx[PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.event2idx[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.event2idx[EOS_TOKEN]

    def encode(self, events: List[str]) -> List[int]:
        """Convert event strings to token ids"""
        return [self.event2idx.get(e, self.event2idx[UNK_TOKEN]) for e in events]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert token ids to event strings"""
        return [self.idx2event.get(i, UNK_TOKEN) for i in ids]

    def save(self, path: str):
        """Save vocabulary to file"""
        data = {
            'event2idx': self.event2idx,
            'idx2event': self.idx2event,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'REMIVocab':
        """Load vocabulary from file"""
        vocab = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab.event2idx = data['event2idx']
        vocab.idx2event = {int(k): v for k, v in data['idx2event'].items()}
        vocab.vocab_size = data['vocab_size']
        vocab.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # Restore token type ranges
        vocab.bar_token = vocab.event2idx['Bar_None']
        vocab.beat_tokens = list(range(vocab.event2idx['Beat_0'], vocab.event2idx['Beat_0'] + 16))

        return vocab

    def __len__(self) -> int:
        return self.vocab_size


def load_existing_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load existing vocabulary from Pop1K7 dataset"""
    with open(vocab_path, 'rb') as f:
        vocab_tuple = pickle.load(f)
    event2idx, idx2event = vocab_tuple
    return event2idx, idx2event


class Pop1K7Vocab:
    """Use existing Pop1K7 vocabulary with added special tokens"""

    def __init__(self, vocab_path: str = None):
        self.special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        if vocab_path:
            self._load_existing(vocab_path)
        else:
            self._build_vocab()

    def _load_existing(self, vocab_path: str):
        """Load and adapt existing Pop1K7 vocabulary"""
        with open(vocab_path, 'rb') as f:
            vocab_tuple = pickle.load(f)
        orig_event2idx, orig_idx2event = vocab_tuple

        # Shift all existing tokens by number of special tokens
        offset = len(self.special_tokens)

        self.event2idx: Dict[str, int] = {}
        self.idx2event: Dict[int, str] = {}

        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.event2idx[token] = i
            self.idx2event[i] = token

        # Add existing tokens with offset
        for event, idx in orig_event2idx.items():
            new_idx = idx + offset
            self.event2idx[event] = new_idx
            self.idx2event[new_idx] = event

        self.vocab_size = len(self.event2idx)

        # Find bar token
        self.bar_token = self.event2idx.get('Bar_None', None)

        # Find pitch tokens range
        pitch_tokens = [idx for event, idx in self.event2idx.items() if event.startswith('Note_Pitch_')]
        self.pitch_token_start = min(pitch_tokens) if pitch_tokens else None
        self.pitch_token_end = max(pitch_tokens) + 1 if pitch_tokens else None

    def _build_vocab(self):
        """Build vocabulary from scratch (fallback)"""
        vocab = REMIVocab()
        self.event2idx = vocab.event2idx
        self.idx2event = vocab.idx2event
        self.vocab_size = vocab.vocab_size
        self.bar_token = vocab.bar_token
        self.pitch_token_start = vocab.pitch_token_start
        self.pitch_token_end = vocab.pitch_token_end

    @property
    def pad_token_id(self) -> int:
        return self.event2idx[PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.event2idx[BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.event2idx[EOS_TOKEN]

    def encode(self, events: List[str]) -> List[int]:
        """Convert event strings to token ids"""
        return [self.event2idx.get(e, self.event2idx[UNK_TOKEN]) for e in events]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert token ids to event strings"""
        return [self.idx2event.get(int(i), UNK_TOKEN) for i in ids]

    def save(self, path: str):
        """Save vocabulary to file"""
        data = {
            'event2idx': self.event2idx,
            'idx2event': self.idx2event,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == '__main__':
    # Test vocabulary
    vocab = REMIVocab()
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"PAD token: {vocab.pad_token_id}")
    print(f"BOS token: {vocab.bos_token_id}")
    print(f"EOS token: {vocab.eos_token_id}")
    print(f"Bar token: {vocab.bar_token}")
    print(f"Pitch token range: {vocab.pitch_token_start}-{vocab.pitch_token_end}")

    # Test existing vocab loading
    existing_vocab_path = '/home/htiuser02/music/music/Pop1K7/Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl'
    pop_vocab = Pop1K7Vocab(existing_vocab_path)
    print(f"\nPop1K7 Vocabulary size: {pop_vocab.vocab_size}")
