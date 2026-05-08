"""
Task 3 — Transformer Music Generator
Usage:
  python src/training/train_transformer.py
  python src/training/train_transformer.py --epochs 5   # quick test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, MAX_SEQ_LEN
from src.models.transformer import MusicTransformer
from src.preprocessing.midi_parser import get_split
from src.preprocessing.tokenizer import build_tokenizer, load_tokenizer, TokenDataset, collate_fn
from src.generation.midi_export import tokens_to_midi

OUTPUTS = Path(__file__).parent.parent.parent / 'outputs'
TOK_DIR = PROCESSED_DIR / 'tokenizer'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Task 3 — MusicTransformer  |  device={device}')

    # tokenizer
    if TOK_DIR.exists():
        try:
            tokenizer = load_tokenizer(TOK_DIR)
            print(f'Loaded tokenizer from {TOK_DIR}')
        except Exception:
            tokenizer = build_tokenizer(save_dir=TOK_DIR)
    else:
        tokenizer = build_tokenizer(save_dir=TOK_DIR)
    vocab_size = len(tokenizer.vocab)
    print(f'vocab_size={vocab_size}')

    # datasets
    train_records = get_split('train')
    val_records   = get_split('validation')
    print(f'Tokenizing train ({len(train_records)} files)...')
    train_ds = TokenDataset(train_records, tokenizer, MAX_SEQ_LEN)
    print(f'Tokenizing val ({len(val_records)} files)...')
    val_ds   = TokenDataset(val_records,   tokenizer, MAX_SEQ_LEN)
    print(f'train={len(train_ds)}  val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    model = MusicTransformer(
        vocab_size      = vocab_size,
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_layers      = args.num_layers,
        dim_feedforward = args.d_model * 4,
        max_seq_len     = MAX_SEQ_LEN,
        dropout         = args.dropout,
    ).to(device)
    print(f'params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir  = OUTPUTS / 'models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'transformer_best.pth'
    best_val  = float('inf')
    train_losses, val_losses, perplexities = [], [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        total = 0.0
        for inp, tgt, attn_mask in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            # attn_mask: True=real → flip to True=ignore for PyTorch
            pad_mask = ~attn_mask.to(device)
            logits = model(inp, padding_mask=pad_mask)
            loss   = MusicTransformer.loss(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg_train = total / len(train_loader)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for inp, tgt, attn_mask in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                pad_mask = ~attn_mask.to(device)
                logits   = model(inp, padding_mask=pad_mask)
                vtotal  += MusicTransformer.loss(logits, tgt).item()
        avg_val = vtotal / len(val_loader)
        ppl     = math.exp(min(avg_val, 20))

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        perplexities.append(ppl)
        scheduler.step()

        print(f'Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  ppl={ppl:.2f}  {time.time()-t0:.1f}s')

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val, 'perplexity': ppl,
                        'vocab_size': vocab_size, 'args': vars(args)}, best_path)
            print(f'  [best] ppl={ppl:.2f}')

    # perplexity report
    plot_dir = OUTPUTS / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'best_val_loss':        best_val,
        'final_perplexity':     perplexities[-1],
        'best_perplexity':      min(perplexities),
        'perplexity_per_epoch': perplexities,
    }
    (plot_dir / 'perplexity_report.json').write_text(json.dumps(report, indent=2))
    print(f'Perplexity report → {plot_dir / "perplexity_report.json"}')

    # loss + perplexity plot
    epochs_range = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, train_losses, label='Train')
    ax1.plot(epochs_range, val_losses,   label='Val', linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Task 3 — Transformer Loss'); ax1.legend()
    ax2.plot(epochs_range, perplexities, color='orange')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity')
    ax2.set_title('Validation Perplexity')
    plt.tight_layout()
    fig.savefig(plot_dir / 'transformer_training_loss.png', dpi=300)
    fig.savefig(plot_dir / 'transformer_training_loss.pdf')
    plt.close(fig)
    print(f'Loss curve → {plot_dir / "transformer_training_loss.png"}')

    # load best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # generate 10 long compositions
    gen_dir = OUTPUTS / 'generated_midis' / 'transformer'
    gen_dir.mkdir(parents=True, exist_ok=True)

    # seed: first 16 real tokens from val set
    seed_inp, _, _ = next(iter(DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)))
    seed = seed_inp[0, :16].unsqueeze(0).to(device)
    # strip any PAD from seed
    seed = seed[:, seed[0] != 0]
    if seed.size(1) == 0:
        seed = torch.tensor([[1]], device=device)

    pad_id = tokenizer.vocab.get('PAD_None', 0)
    generated_ok = 0
    for i in range(10):
        tokens = model.generate(seed, max_new_tokens=512,
                                temperature=args.temperature, top_k=args.top_k)
        token_list = [t for t in tokens[0].cpu().tolist() if t != pad_id]
        out_path   = str(gen_dir / f'sample_{i+1}.mid')
        result     = tokens_to_midi(token_list, tokenizer, output_path=out_path)
        if result is not None:
            generated_ok += 1
            print(f'  Generated → {out_path}')
        else:
            print(f'  [skip] sample_{i+1} — token decode failed')

    print(f'\nDone. Best val loss: {best_val:.4f}  best ppl: {min(perplexities):.2f}')
    print(f'Generated {generated_ok}/10 MIDI files')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--d-model',     type=int,   default=256)
    p.add_argument('--nhead',       type=int,   default=8)
    p.add_argument('--num-layers',  type=int,   default=6)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=1.1)
    p.add_argument('--top-k',       type=int,   default=50)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
