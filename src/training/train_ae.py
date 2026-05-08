"""
Task 1 — LSTM Autoencoder
Usage:
  python src/training/train_ae.py
  python src/training/train_ae.py --epochs 5   # quick test
"""
import argparse
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
from src.config import PROCESSED_DIR, SEQ_LEN, BATCH_SIZE
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.piano_roll import PianoRollDataset
from src.generation.midi_export import roll_to_midi, verify_midi

OUTPUTS = Path(__file__).parent.parent.parent / 'outputs'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Task 1 — LSTM Autoencoder  |  device={device}')

    # datasets
    train_ds = PianoRollDataset(PROCESSED_DIR / 'train_piano_rolls.npy')
    val_ds   = PianoRollDataset(PROCESSED_DIR / 'val_piano_rolls.npy')
    print(f'train={len(train_ds)}  val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    model = LSTMAutoencoder(
        hidden_dim  = args.hidden_dim,
        latent_dim  = args.latent_dim,
        num_layers  = args.num_layers,
        dropout     = args.dropout,
    ).to(device)
    print(f'params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir  = OUTPUTS / 'models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'ae_best.pth'
    best_val  = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        total = 0.0
        for x in train_loader:
            x      = x.to(device)
            logits = model(x)
            loss   = LSTMAutoencoder.focal_loss(logits, x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg_train = total / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                val_total += LSTMAutoencoder.focal_loss(model(x), x).item()
        avg_val = val_total / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f'Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  {time.time()-t0:.1f}s')

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val, 'args': vars(args)}, best_path)
            print(f'  [best] val={best_val:.4f}')

    # loss curve
    plot_dir = OUTPUTS / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses,   label='Val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Focal Loss')
    ax.set_title('Task 1 — LSTM Autoencoder')
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / 'ae_training_loss.png', dpi=300)
    fig.savefig(plot_dir / 'ae_training_loss.pdf')
    plt.close(fig)
    print(f'Loss curve → {plot_dir / "ae_training_loss.png"}')

    # generate 5 MIDI samples
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    gen_dir = OUTPUTS / 'generated_midis' / 'ae'
    gen_dir.mkdir(parents=True, exist_ok=True)

    rolls = model.generate(n=5, seq_len=SEQ_LEN, device=device)  # (5, T, 88)
    for i, roll in enumerate(rolls):
        path = str(gen_dir / f'sample_{i+1}.mid')
        roll_to_midi(roll, output_path=path)
        verify_midi(path)
        print(f'  Generated → {path}')

    print(f'\nDone. Best val loss: {best_val:.4f}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch-size', type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--hidden-dim', type=int,   default=256)
    p.add_argument('--latent-dim', type=int,   default=64)
    p.add_argument('--num-layers', type=int,   default=2)
    p.add_argument('--dropout',    type=float, default=0.3)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
