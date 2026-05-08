from pathlib import Path

ROOT        = Path(__file__).parent.parent
MAESTRO_DIR = ROOT / "Dataset" / "maestro-v3.0.0-midi" / "maestro-v3.0.0"

PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR     = ROOT / "Dataset" / "train_test_split"

# piano roll
FS           = 16
PIANO_ROLL_FS = FS   # alias used by notebooks
SEQ_LEN      = 128
N_PITCHES    = 88
PITCH_LOW    = 21   # A0
PITCH_HIGH   = 108  # C8

# tokenizer
N_VELOCITY_BINS = 32
MAX_SEQ_LEN     = 512

# training
BATCH_SIZE = 64
