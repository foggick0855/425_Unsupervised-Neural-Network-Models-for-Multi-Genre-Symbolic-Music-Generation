import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FS, SEQ_LEN, N_PITCHES, PITCH_LOW, PITCH_HIGH


def midi_to_roll(path, fs=FS):
    pm   = pretty_midi.PrettyMIDI(path)
    roll = pm.get_piano_roll(fs=fs)        # (128, T)
    roll = roll[PITCH_LOW : PITCH_HIGH + 1]  # (88, T)
    roll = roll.T                           # (T, 88)
    roll = (roll > 0).astype(np.float32)   # binarize
    return roll


def segment_roll(roll, seq_len=SEQ_LEN, min_active_ratio=0.02):
    min_active = int(seq_len * N_PITCHES * min_active_ratio)
    windows = []
    T = roll.shape[0]
    for start in range(0, T - seq_len + 1, seq_len):
        w = roll[start : start + seq_len]
        if w.sum() >= min_active:
            windows.append(w)
    return windows


def build_npy(records, out_path, seq_len=SEQ_LEN):
    windows = []
    total = len(records)
    for i, rec in enumerate(records):
        try:
            roll = midi_to_roll(rec["path"])
            windows.extend(segment_roll(roll, seq_len))
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}  windows: {len(windows)}")
    arr = np.array(windows, dtype=np.float32)   # (N, 128, 88)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"  saved {arr.shape} → {out_path}")
    return arr


class PianoRollDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)   # (N, 128, 88)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()
