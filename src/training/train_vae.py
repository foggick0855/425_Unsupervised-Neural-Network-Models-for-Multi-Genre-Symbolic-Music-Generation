"""
Task 2 — MusicVAE
Usage:
  python src/training/train_vae.py
  python src/training/train_vae.py --epochs 5   # quick test
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
from src.models.vae import MusicVAE
from src.preprocessing.piano_roll import PianoRollDataset
from src.generation.midi_export import roll_to_midi, verify_midi

OUTPUTS = Path(__file__).parent.parent.parent / 'outputs'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Task 2 — MusicVAE  |  device={device}  beta_max={args.beta_max}  warmup={args.warmup}')

    train_ds = PianoRollDataset(PROCESSED_DIR / 'train_piano_rolls.npy')
    val_ds   = PianoRollDataset(PROCESSED_DIR / 'val_piano_rolls.npy')
    print(f'train={len(train_ds)}  val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MusicVAE(
        hidden_dim  = args.hidden_dim,
        latent_dim  = args.latent_dim,
        num_layers  = args.num_layers,
        dropout     = args.dropout,
    ).to(device)
    print(f'params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir  = OUTPUTS / 'models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'vae_best.pth'
    best_val  = float('inf')

    train_totals, train_recons, train_kls = [], [], []
    val_totals,   val_recons,   val_kls   = [], [], []

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / max(args.warmup, 1))
        t0   = time.time()

        model.train()
        tot, rec, kld = 0.0, 0.0, 0.0
        for x in train_loader:
            x               = x.to(device)
            logits, mu, lv  = model(x)
            recon = MusicVAE.focal_loss(logits, x)
            kl    = MusicVAE.kl_loss(mu, lv)
            loss  = recon + beta * kl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); rec += recon.item(); kld += kl.item()

        n = len(train_loader)
        train_totals.append(tot / n)
        train_recons.append(rec / n)
        train_kls.append(kld / n)

        model.eval()
        vtot, vrec, vkl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x in val_loader:
                x              = x.to(device)
                logits, mu, lv = model(x)
                recon = MusicVAE.focal_loss(logits, x)
                kl    = MusicVAE.kl_loss(mu, lv)
                vtot += (recon + beta * kl).item()
                vrec += recon.item()
                vkl  += kl.item()

        vn = len(val_loader)
        val_totals.append(vtot / vn)
        val_recons.append(vrec / vn)
        val_kls.append(vkl / vn)

        print(f'Epoch {epoch:3d}/{args.epochs}  beta={beta:.3f}  '
              f'train={tot/n:.4f}(r={rec/n:.4f} kl={kld/n:.4f})  '
              f'val={vtot/vn:.4f}  {time.time()-t0:.1f}s')

        if vtot / vn < best_val:
            best_val = vtot / vn
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val, 'args': vars(args)}, best_path)
            print(f'  [best] val={best_val:.4f}')

    # loss curves — 3 panels
    plot_dir = OUTPUTS / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, tr, vl, title in zip(
        axes,
        [train_totals, train_recons, train_kls],
        [val_totals,   val_recons,   val_kls],
        ['Total Loss', 'Recon Loss', 'KL Loss'],
    ):
        ax.plot(epochs_range, tr, label='Train')
        ax.plot(epochs_range, vl, label='Val', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend()
    plt.suptitle('Task 2 — MusicVAE')
    plt.tight_layout()
    fig.savefig(plot_dir / 'vae_training_loss.png', dpi=300)
    fig.savefig(plot_dir / 'vae_training_loss.pdf')
    plt.close(fig)
    print(f'Loss curve → {plot_dir / "vae_training_loss.png"}')

    # load best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # generate 8 MIDI samples
    gen_dir = OUTPUTS / 'generated_midis' / 'vae'
    gen_dir.mkdir(parents=True, exist_ok=True)

    kept    = []
    attempt = 0
    while len(kept) < 8 and attempt < 300:
        attempt += 1
        rolls = model.generate(n=16, seq_len=SEQ_LEN, device=device)
        for roll in rolls:
            if len(kept) >= 8:
                break
            tmp = str(gen_dir / f'_tmp_{attempt}.mid')
            roll_to_midi(roll, output_path=tmp)
            if verify_midi(tmp, min_notes=50, min_duration=5.0):
                kept.append(roll)

    for i, roll in enumerate(kept):
        path = str(gen_dir / f'sample_{i+1}.mid')
        roll_to_midi(roll, output_path=path)
        print(f'  Generated → {path}')
    for f in gen_dir.glob('_tmp_*.mid'):
        f.unlink()

    # latent interpolation — 8 steps, α ∈ {0, 1/7, ..., 1}
    print('Running latent interpolation (8 steps)...')
    x1 = val_ds[0].to(device)
    x2 = val_ds[len(val_ds) // 2].to(device)
    interp_rolls = model.interpolate(x1, x2, steps=8, seq_len=SEQ_LEN)
    for i, roll in enumerate(interp_rolls):
        path = str(gen_dir / f'interp_{i+1:02d}.mid')
        roll_to_midi(roll, output_path=path)
        print(f'  Interp {i+1}/8 → {path}')

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
    p.add_argument('--beta-max',   type=float, default=0.5)
    p.add_argument('--warmup',     type=int,   default=50)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
