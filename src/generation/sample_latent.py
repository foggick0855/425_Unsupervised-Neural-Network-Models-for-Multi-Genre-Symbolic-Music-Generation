"""
sample_latent.py — VAE latent space sampler.

Loads a trained MusicVAE checkpoint, samples z ~ N(0, I), decodes each
sample to a binary piano-roll, and exports the results as MIDI files.

Usage:
    python src/generation/sample_latent.py
    python src/generation/sample_latent.py --n 8 --checkpoint outputs/models/vae_best.pth
    python src/generation/sample_latent.py --interpolate --checkpoint outputs/models/vae_best.pth
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import N_PITCHES, SEQ_LEN
from src.models.vae import MusicVAE
from src.preprocessing.piano_roll import piano_roll_to_midi

OUTPUTS = Path(__file__).parent.parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device) -> MusicVAE:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    args  = ckpt.get("args", {})
    model = MusicVAE(
        input_size  = N_PITCHES,
        hidden_size = args.get("hidden_size", 512),
        latent_dim  = args.get("latent_dim",  256),
        num_layers  = args.get("num_layers",  2),
        seq_len     = SEQ_LEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def sample_and_export(
    model:     MusicVAE,
    n:         int,
    device:    torch.device,
    out_dir:   Path,
    temperature: float = 1.0,
    threshold:   float = 0.3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rolls = model.generate(n_samples=n, device=device,
                           temperature=temperature, threshold=threshold)
    rolls = rolls.cpu().numpy().transpose(0, 2, 1)   # (n, 88, T)
    for i, roll in enumerate(rolls):
        path = out_dir / f"latent_sample_{i+1}.mid"
        piano_roll_to_midi(roll, output_path=str(path))
        print(f"  Saved -> {path}")


def interpolate_and_export(
    model:   MusicVAE,
    device:  torch.device,
    out_dir: Path,
    steps:   int = 8,
) -> None:
    """
    Sample two points z1, z2 ~ N(0, I) and interpolate linearly
    between them in latent space, exporting each step as a MIDI file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    z1 = torch.randn(1, model.latent_dim, device=device)
    z2 = torch.randn(1, model.latent_dim, device=device)

    alphas = torch.linspace(0, 1, steps, device=device)
    for i, alpha in enumerate(alphas):
        z     = (1 - alpha) * z1 + alpha * z2
        logit = model.decoder(z)
        roll  = torch.bernoulli(torch.sigmoid(logit))   # (1, T, 88)
        roll  = roll[0].cpu().numpy().T                 # (88, T)
        path  = out_dir / f"interp_{i+1:02d}_alpha{alpha:.2f}.mid"
        piano_roll_to_midi(roll, output_path=str(path))
        print(f"  Interpolation step {i+1}/{steps} (α={alpha:.2f}) -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample VAE latent space")
    p.add_argument("--checkpoint",   default="outputs/models/vae_best.pth",
                   help="Path to trained VAE checkpoint (.pth)")
    p.add_argument("--n",            type=int,   default=8,
                   help="Number of samples to generate")
    p.add_argument("--temperature",  type=float, default=1.0,
                   help="Sampling temperature (higher = more random)")
    p.add_argument("--threshold",    type=float, default=0.3,
                   help="Binarization threshold (default 0.3, below 0.5 compensates for note underestimation)")
    p.add_argument("--interpolate",  action="store_true",
                   help="Run latent interpolation between two random samples instead")
    p.add_argument("--steps",        type=int,   default=8,
                   help="Number of interpolation steps (used with --interpolate)")
    p.add_argument("--out-dir",      default=None,
                   help="Output directory (default: outputs/generated_midis/latent_samples/)")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = load_model(args.checkpoint, device)
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS / "generated_midis" / "latent_samples"

    if args.interpolate:
        print(f"\nRunning latent interpolation ({args.steps} steps)...")
        interpolate_and_export(model, device, out_dir / "interpolation", steps=args.steps)
    else:
        print(f"\nSampling {args.n} points from N(0, I)...")
        sample_and_export(model, args.n, device, out_dir,
                          temperature=args.temperature, threshold=args.threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
