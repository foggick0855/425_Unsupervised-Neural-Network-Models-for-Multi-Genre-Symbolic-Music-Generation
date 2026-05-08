from pathlib import Path
from typing import List, Optional, Union
import sys

import numpy as np
import pretty_midi
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FS, PITCH_LOW


def roll_to_midi(
    roll: Union[np.ndarray, torch.Tensor],
    fs: int = FS,
    pitch_low: int = PITCH_LOW,
    velocity: int = 80,
    tempo: float = 120.0,
    output_path: Optional[str] = None,
) -> pretty_midi.PrettyMIDI:
    if isinstance(roll, torch.Tensor):
        roll = roll.cpu().numpy()

    # normalise to (88, T)
    if roll.ndim == 2 and roll.shape[0] != 88:
        roll = roll.T
    roll = (roll > 0.5).astype(np.uint8)

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    dt   = 1.0 / fs

    n_pitches, T = roll.shape
    for p in range(n_pitches):
        note_on = None
        for t in range(T):
            if roll[p, t] and note_on is None:
                note_on = t
            elif not roll[p, t] and note_on is not None:
                inst.notes.append(pretty_midi.Note(velocity, p + pitch_low, note_on * dt, t * dt))
                note_on = None
        if note_on is not None:
            inst.notes.append(pretty_midi.Note(velocity, p + pitch_low, note_on * dt, T * dt))

    midi.instruments.append(inst)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        midi.write(output_path)
    return midi


def tokens_to_midi(
    token_ids: List[int],
    tokenizer,
    output_path: Optional[str] = None,
) -> Optional[pretty_midi.PrettyMIDI]:
    try:
        from miditok import TokSequence
        tok_seq = TokSequence(ids=token_ids)
        score   = tokenizer.decode([tok_seq])
        path    = output_path or "decoded.mid"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        score.dump_midi(path)
        return pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(f"tokens_to_midi failed: {e}")
        return None


def verify_midi(path: str, min_notes: int = 50, min_duration: float = 5.0) -> bool:
    try:
        pm    = pretty_midi.PrettyMIDI(path)
        notes = [n for inst in pm.instruments for n in inst.notes]
        if len(notes) < min_notes:
            print(f"  [skip] {path}: {len(notes)} notes < {min_notes}")
            return False
        if pm.get_end_time() < min_duration:
            print(f"  [skip] {path}: {pm.get_end_time():.1f}s < {min_duration}s")
            return False
        return True
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return False
