import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import MAESTRO_DIR


def load_maestro(max_files=None):
    df = pd.read_csv(MAESTRO_DIR / "maestro-v3.0.0.csv")
    records = []
    for _, row in df.iterrows():
        records.append({
            "path":     str(MAESTRO_DIR / row["midi_filename"]),
            "split":    row["split"],          # "train" | "validation" | "test"
            "composer": row["canonical_composer"],
            "title":    row["canonical_title"],
            "duration": row["duration"],
            "year":     int(row["year"]),
        })
    return records[:max_files] if max_files else records


def get_split(split_name):
    return [r for r in load_maestro() if r["split"] == split_name]
