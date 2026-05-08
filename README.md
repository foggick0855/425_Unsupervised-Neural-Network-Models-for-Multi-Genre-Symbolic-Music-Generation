# Music Generation with Unsupervised Neural Networks

Symbolic piano music generation using three progressively complex unsupervised deep learning models trained on the MAESTRO v3.0.0 dataset.

**Course:** CSE 425 — Machine Learning  
**Institution:** BRAC University

---

## Team Members & Contributions

| Member | Contribution |
|--------|-------------|
| **Mahadi Hasan Fahim** | EDA, Preprocessing pipeline, Task 1 (LSTM Autoencoder) |
| **Mubassir Raiyan** | Task 2 (Variational Autoencoder), Baseline models |
| **Tanjim Rahaman Fardin** | Task 3 (Transformer), Report writing |

---

## Generated MIDI Files

All generated MIDI samples are available for listening:

**[Google Drive — Generated MIDI Files](https://drive.google.com/drive/folders/1n7gyIyUzjG9LVD5ZzIh4z1M-wgBidOEe?usp=sharing)**

| Folder | Contents |
|--------|----------|
| `ae/` | 5 samples from LSTM Autoencoder (Task 1) |
| `vae/` | 8 random samples + 8 latent interpolation steps (Task 2) |
| `transformer/` | 10 long-form compositions (Task 3) |
| `baselines/` | Random generator, Markov 1st-order, Markov 2nd-order |

---

## Tasks

| Task | Model | Input | Output |
|------|-------|-------|--------|
| 1 | LSTM Autoencoder | Binary piano-roll (128×88) | 5 generated MIDI samples |
| 2 | Variational Autoencoder | Binary piano-roll (128×88) | 8 samples + 8 latent interpolations |
| 3 | Transformer (GPT-style) | REMI token sequences (512 tokens) | 10 long-form compositions |

---

## Project Structure

```
Project/
├── README.md
├── requirements.txt
├── SETUP.md
│
├── Dataset/
│   └── maestro-v3.0.0-midi/         # MAESTRO v3 MIDI files (1,276 recordings)
│       └── maestro-v3.0.0/
│           └── 2004–2018/           # Year-organised MIDI files
│
├── data/
│   └── processed/
│       ├── train_piano_rolls.npy    # Preprocessed piano-roll windows (N, 128, 88)
│       ├── val_piano_rolls.npy
│       ├── test_piano_rolls.npy
│       └── tokenizer/
│           └── tokenizer.json       # Saved REMI tokenizer (vocab=284)
│
├── notebooks/
│   ├── preprocessing.ipynb          # EDA + piano-roll pipeline walkthrough
│   └── baseline_markov.ipynb        # Random generator + Markov chain baselines
│
├── src/
│   ├── config.py                    # Global constants (FS=16, SEQ_LEN=128, N_PITCHES=88)
│   │
│   ├── preprocessing/
│   │   ├── midi_parser.py           # MAESTRO CSV loader, train/val/test split builder
│   │   ├── piano_roll.py            # midi_to_roll, segment_roll, PianoRollDataset
│   │   └── tokenizer.py             # REMI tokenizer build/load, TokenDataset, collate_fn
│   │
│   ├── models/
│   │   ├── autoencoder.py           # Task 1 — LSTM Autoencoder (Focal Loss, 1.8M params)
│   │   ├── vae.py                   # Task 2 — MusicVAE (KL annealing, latent interpolation)
│   │   └── transformer.py           # Task 3 — Decoder-only Transformer (sinusoidal PE)
│   │
│   ├── training/
│   │   ├── train_ae.py              # Task 1 training script (50 epochs, Adam lr=1e-3)
│   │   ├── train_vae.py             # Task 2 training script (beta annealing warmup)
│   │   └── train_transformer.py     # Task 3 training script (top-k=50, temp=1.1)
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # evaluate_midi, compare_models, print_report
│   │   ├── pitch_histogram.py       # Pitch histogram L1 distance (spec eq.)
│   │   ├── rhythm_score.py          # Rhythm diversity score (unique durations / total)
│   │   └── compare_all.py           # Unified comparison table (Tasks 1–3 + baselines)
│   │
│   └── generation/
│       ├── midi_export.py           # roll_to_midi, tokens_to_midi, verify_midi
│       ├── generate_music.py        # Unified generation CLI for all models
│       └── sample_latent.py         # VAE latent space sampler
│
├── outputs/
│   ├── generated_midis/
│   │   ├── ae/                      # sample_1–5.mid + .wav
│   │   ├── vae/                     # sample_1–8.mid, interp_01–08.mid
│   │   ├── transformer/             # sample_1–10.mid + .wav
│   │   └── baselines/
│   │       ├── random/              # random_1–5.mid
│   │       ├── markov_order1/       # markov1_1–5.mid
│   │       └── markov_order2/       # markov2_1–5.mid
│   │
│   ├── models/
│   │   ├── ae_best.pth              # Best AE checkpoint (val_loss=0.0986, epoch 50)
│   │   ├── vae_best.pth             # Best VAE checkpoint
│   │   └── transformer_best.pth     # Best Transformer checkpoint
│   │
│   └── plots/
│       ├── ae_training_loss.png/.pdf
│       ├── vae_training_loss.png/.pdf
│       ├── transformer_training_loss.png/.pdf
│       ├── comparison_table.png/.pdf
│       ├── baseline_comparison.png
│       ├── baseline_metrics.json
│       ├── evaluation_report.json   # Per-model averaged metrics
│       ├── perplexity_report.json   # Transformer perplexity
│       └── eda_*.png/.pdf           # EDA plots (pitch, duration, velocity, note count)
│
└── report/
    ├── conference_101719.tex        # IEEE conference paper (LaTeX source)
    ├── IEEEtran.cls                 # IEEE style class
    └── figures/                     # Plots copied from outputs/plots/ for LaTeX
        ├── ae_training_loss.png
        ├── vae_training_loss.png
        ├── transformer_training_loss.png
        ├── comparison_table.png
        ├── baseline_comparison.png
        ├── eda_pitch.png
        └── eda_duration.png
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. Install PyTorch (CUDA build — adjust cu128 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import torch, pretty_midi, miditok; print('OK')"
```

---

## Train from Scratch

```bash
# Build piano-roll dataset from MAESTRO
python src/preprocessing/build_dataset.py

# Task 1 — LSTM Autoencoder
python src/training/train_ae.py --epochs 50

# Task 2 — Variational Autoencoder
python src/training/train_vae.py --epochs 50 --beta-max 0.5 --warmup 50

# Task 3 — Transformer
python src/training/train_transformer.py --epochs 20
```

---

## Evaluate

```bash
# Per-model metrics (rhythm diversity, repetition ratio, pitch histogram distance)
python src/evaluation/metrics.py

# Full comparison table (Tasks 1–3 vs baselines) → outputs/plots/comparison_table.png
python src/evaluation/compare_all.py
```

---

## Key Results

| Model | Val Loss | Rhythm Div. | Pitch Hist. Dist. |
|-------|----------|-------------|-------------------|
| Random Baseline | — | 0.072 | 0.423 |
| Markov 2nd-order | — | 0.131 | 0.323 |
| Task 1: LSTM AE | 0.0986 | **0.592** | 0.444 |
| Task 2: VAE | — | 0.073 | **0.233** |
| Task 3: Transformer | — | 0.112 | 0.566 |

> Rhythm diversity: higher is better. Pitch histogram distance: lower is better (closer to real MAESTRO).

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x recommended (CPU fallback supported)
- ~4 GB disk for model checkpoints and generated files
- MAESTRO v3.0.0 dataset placed at `Dataset/maestro-v3.0.0-midi/maestro-v3.0.0/`
