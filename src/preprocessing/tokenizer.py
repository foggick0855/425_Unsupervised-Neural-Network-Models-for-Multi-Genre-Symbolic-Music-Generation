from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import N_VELOCITY_BINS, MAX_SEQ_LEN


def build_tokenizer(save_dir=None):
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        num_velocities=N_VELOCITY_BINS,
        use_chords=False,
        use_programs=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        beat_res={(0, 4): 8, (4, 12): 4},
    )
    tok = REMI(config)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(save_dir))
    return tok


def load_tokenizer(save_dir):
    from miditok import REMI
    return REMI.from_pretrained(str(save_dir))


def tokenize_file(path, tokenizer, max_seq_len=MAX_SEQ_LEN, min_len=32):
    try:
        import symusic
        score = symusic.Score(path)
        raw   = tokenizer(score)
        seqs  = []
        for tok_seq in raw:
            ids = tok_seq.ids
            if len(ids) < min_len:
                continue
            for start in range(0, len(ids), max_seq_len):
                chunk = ids[start : start + max_seq_len]
                if len(chunk) >= min_len:
                    seqs.append(chunk)
        return seqs or None
    except Exception:
        return None


def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(t.size(0) for t in inputs)
    B = len(inputs)
    inp_pad  = torch.zeros(B, max_len, dtype=torch.long)
    tgt_pad  = torch.full((B, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.size(0)
        inp_pad[i, :L]   = inp
        tgt_pad[i, :L]   = tgt
        attn_mask[i, :L] = True
    return inp_pad, tgt_pad, attn_mask


class TokenDataset(Dataset):
    def __init__(self, records, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.pad_id      = tokenizer["PAD_None"]
        self.max_seq_len = max_seq_len
        self.samples: List[List[int]] = []
        for rec in records:
            seqs = tokenize_file(rec["path"], tokenizer, max_seq_len)
            if seqs:
                self.samples.extend(seqs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        else:
            ids = ids[: self.max_seq_len]
        t = torch.tensor(ids, dtype=torch.long)
        return t[:-1], t[1:]
