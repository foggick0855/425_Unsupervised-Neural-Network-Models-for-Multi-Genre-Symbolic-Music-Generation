"""
Unified generation script — load any trained model and generate MIDI files.

Usage:
  # Generate from AE checkpoint
  python src/generation/generate_music.py --model ae --n 5

  # Generate from VAE checkpoint
  python src/generation/generate_music.py --model vae --n 8 --temperature 1.2

  # Generate from Transformer checkpoint
  python src/generation/generate_music.py --model transformer --n 10 --max-tokens 512

  # Generate from RLHF checkpoint
  python src/generation/generate_music.py --model rlhf --n 5
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import N_PITCHES, SEQ_LEN, MAX_TOKEN_SEQ_LEN, PROCESSED_DIR
from src.preprocessing.piano_roll import piano_roll_to_midi

OUTPUTS = Path(__file__).parent.parent.parent / "outputs"


# ---------------------------------------------------------------------------
# AE / VAE generation (piano-roll models)
# ---------------------------------------------------------------------------
def generate_ae(args: argparse.Namespace, device: torch.device) -> None:
    from src.models.autoencoder import LSTMAutoencoder

    ckpt_path = OUTPUTS / "models" / "autoencoder_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No AE checkpoint at {ckpt_path} — run train_ae.py first.")

    ckpt      = torch.load(ckpt_path, map_location=device)
    saved     = ckpt.get("args", {})

    model = LSTMAutoencoder(
        input_size  = N_PITCHES,
        hidden_size = saved.get("hidden_size", 512),
        latent_dim  = saved.get("latent_dim",  256),
        num_layers  = saved.get("num_layers",  2),
        seq_len     = SEQ_LEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    out_dir = OUTPUTS / "generated_midis" / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)

    rolls = model.generate(args.n, device, temperature=args.temperature)  # (n, T, 88)
    rolls = rolls.cpu().numpy().transpose(0, 2, 1)                         # (n, 88, T)
    for i, roll in enumerate(rolls):
        path = str(out_dir / f"generated_{i+1:02d}.mid")
        piano_roll_to_midi(roll, output_path=path)
        print(f"Saved -> {path}")


def generate_vae(args: argparse.Namespace, device: torch.device) -> None:
    from src.models.vae import MusicVAE

    ckpt_path = OUTPUTS / "models" / "vae_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No VAE checkpoint at {ckpt_path} — run train_vae.py first.")

    ckpt  = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})

    model = MusicVAE(
        input_size  = N_PITCHES,
        hidden_size = saved.get("hidden_size", 512),
        latent_dim  = saved.get("latent_dim",  256),
        num_layers  = saved.get("num_layers",  2),
        seq_len     = SEQ_LEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    out_dir = OUTPUTS / "generated_midis" / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)

    rolls = model.generate(args.n, device, temperature=args.temperature)
    rolls = rolls.cpu().numpy().transpose(0, 2, 1)
    for i, roll in enumerate(rolls):
        path = str(out_dir / f"generated_{i+1:02d}.mid")
        piano_roll_to_midi(roll, output_path=path)
        print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# Transformer generation (token-based)
# ---------------------------------------------------------------------------
def generate_transformer(args: argparse.Namespace, device: torch.device, rlhf: bool = False) -> None:
    from src.models.transformer import MusicTransformer
    from src.preprocessing.tokenizer import build_tokenizer, load_tokenizer, tokenize_file, tokens_to_midi

    tok_dir   = PROCESSED_DIR / "tokenizer"
    tokenizer = load_tokenizer(tok_dir) if tok_dir.exists() else build_tokenizer(tok_dir)
    vocab_size = len(tokenizer.vocab)

    ckpt_name = "transformer_rlhf.pth" if rlhf else "transformer_best.pth"
    ckpt_path = OUTPUTS / "models" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}.")

    ckpt  = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("args", {})

    model = MusicTransformer(
        vocab_size      = vocab_size,
        d_model         = saved.get("d_model",    256),
        nhead           = saved.get("nhead",       8),
        num_layers      = saved.get("num_layers",  6),
        dim_feedforward = saved.get("d_model", 256) * 4,
        max_seq_len     = MAX_TOKEN_SEQ_LEN,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    subdir  = "rlhf" if rlhf else "transformer"
    out_dir = OUTPUTS / "generated_midis" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a short prompt from the first training file
    try:
        records = json.loads((OUTPUTS.parent / "Dataset" / "train_test_split" / "train.json").read_text())[:1]
        seqs    = tokenize_file(records[0]["path"], tokenizer, max_seq_len=32)
        prompt_ids = seqs[0][:16] if seqs else [1]
    except Exception:
        prompt_ids = [1]

    pad_id = tokenizer.vocab.get("PAD_None", 0)

    for i in range(args.n):
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        tokens = model.generate(prompt, max_new_tokens=args.max_tokens,
                                temperature=args.temperature, top_k=50)
        ids    = [t for t in tokens[0].cpu().tolist() if t != pad_id]
        path   = str(out_dir / f"generated_{i+1:02d}.mid")
        try:
            tokens_to_midi(ids, tokenizer, output_path=path)
            print(f"Saved -> {path}")
        except Exception as e:
            print(f"Could not save sample {i+1}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI from a trained model")
    p.add_argument("--model", type=str, default="ae",
                   choices=["ae", "vae", "transformer", "rlhf"],
                   help="Which model to generate from")
    p.add_argument("--n",           type=int,   default=5,   help="Number of samples")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--max-tokens",  type=int,   default=512, help="Transformer max new tokens")
    p.add_argument("--device",      type=str,   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Generating {args.n} samples from model='{args.model}' on {device}")

    if args.model == "ae":
        generate_ae(args, device)
    elif args.model == "vae":
        generate_vae(args, device)
    elif args.model == "transformer":
        generate_transformer(args, device, rlhf=False)
    elif args.model == "rlhf":
        generate_transformer(args, device, rlhf=True)
