import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MAX_SEQ_LEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_seq_len, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        self.fc_out      = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, padding_mask=None):
        # tokens: (B, T) long  |  padding_mask: (B, T) bool True=ignore
        B, T = tokens.shape
        x    = self.token_emb(tokens) * math.sqrt(self.d_model)
        x    = self.pos_enc(x)
        # bool causal mask: True = cannot attend (upper-triangle = future positions)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device).bool()
        out  = self.transformer(x, mask=causal, src_key_padding_mask=padding_mask, is_causal=True)
        return self.fc_out(out)   # (B, T, vocab_size)

    @staticmethod
    def loss(logits, targets, ignore_index=-100):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, temperature=1.1, top_k=50):
        # prompt: (1, T_seed) long
        self.eval()
        x = prompt.clone()
        for _ in range(max_new_tokens):
            x_in   = x[:, -self.max_seq_len:]
            logits = self(x_in)[:, -1, :] / max(temperature, 1e-6)
            logits[:, 0] = float('-inf')   # suppress PAD token
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float('-inf')
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            x = torch.cat([x, next_tok], dim=1)
        return x   # (1, T_seed + max_new_tokens)
