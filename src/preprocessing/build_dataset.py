"""
build_dataset.py — Run once to create processed metadata and splits.

Outputs (inside Dataset/processed/ and Dataset/train_test_split/):
  processed/maestro_metadata.json
  processed/lakh_metadata.json
  processed/groove_metadata.json
  processed/all_metadata.json
  train_test_split/train.json
  train_test_split/val.json
  train_test_split/test.json

Usage (from Project/ root with venv activated):
  python src/preprocessing/build_dataset.py

Optional flags:
  --sample N     Only process N files per dataset (quick test)
  --validate     Try loading every MIDI file (slow but thorough)
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Make src importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import PROCESSED_DIR, SPLIT_DIR
from src.preprocessing.midi_parser import load_maestro, load_lakh, load_groove

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation (optional)
# ---------------------------------------------------------------------------
def validate_midi(path: str) -> bool:
    """Return True if pretty_midi can load the file and it has notes."""
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(path)
        total = sum(len(i.notes) for i in midi.instruments)
        return total >= 4
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------
def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} records -> {path}")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def print_stats(records: list[dict], label: str = "") -> None:
    split_counts  = Counter(r["split"]  for r in records)
    genre_counts  = Counter(r["genre"]  for r in records)
    dataset_counts = Counter(r["dataset"] for r in records)
    print(f"\n{'='*50}")
    print(f"  {label}  ({len(records)} total files)")
    print(f"{'='*50}")
    print(f"  Splits:   {dict(split_counts)}")
    print(f"  Genres:   {dict(genre_counts)}")
    print(f"  Datasets: {dict(dataset_counts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build preprocessed metadata for all datasets.")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Max files per dataset (for quick testing).")
    parser.add_argument("--validate", action="store_true",
                        help="Validate every MIDI file with pretty_midi (slow).")
    parser.add_argument("--no-lakh",  action="store_true", help="Skip Lakh dataset.")
    args = parser.parse_args()

    logger.info("=== Building dataset metadata ===")
    logger.info(f"sample={args.sample}  validate={args.validate}")

    # ------------------------------------------------------------------
    # 1. Load file lists from all datasets
    # ------------------------------------------------------------------
    maestro_records = load_maestro(args.sample)
    groove_records  = load_groove(args.sample)
    lakh_records    = [] if args.no_lakh else load_lakh(args.sample)

    # ------------------------------------------------------------------
    # 2. Optional validation pass
    # ------------------------------------------------------------------
    def maybe_validate(records: list[dict], name: str) -> list[dict]:
        if not args.validate:
            return records
        logger.info(f"Validating {name} ({len(records)} files)…")
        valid = []
        for i, rec in enumerate(records):
            if validate_midi(rec["path"]):
                valid.append(rec)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(records)} checked, {len(valid)} valid so far")
        logger.info(f"{name}: {len(valid)}/{len(records)} files passed validation")
        return valid

    maestro_records = maybe_validate(maestro_records, "MAESTRO")
    groove_records  = maybe_validate(groove_records,  "Groove")
    lakh_records    = maybe_validate(lakh_records,    "Lakh")

    # ------------------------------------------------------------------
    # 3. Save per-dataset metadata
    # ------------------------------------------------------------------
    save_json(maestro_records, PROCESSED_DIR / "maestro_metadata.json")
    save_json(groove_records,  PROCESSED_DIR / "groove_metadata.json")
    save_json(lakh_records,    PROCESSED_DIR / "lakh_metadata.json")

    # ------------------------------------------------------------------
    # 4. Combine and create splits
    # ------------------------------------------------------------------
    all_records = maestro_records + groove_records + lakh_records
    save_json(all_records, PROCESSED_DIR / "all_metadata.json")

    train = [r for r in all_records if r["split"] == "train"]
    val   = [r for r in all_records if r["split"] == "val"]
    test  = [r for r in all_records if r["split"] == "test"]

    save_json(train, SPLIT_DIR / "train.json")
    save_json(val,   SPLIT_DIR / "val.json")
    save_json(test,  SPLIT_DIR / "test.json")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print_stats(maestro_records, "MAESTRO")
    print_stats(groove_records,  "Groove")
    print_stats(lakh_records,    "Lakh (clean_midi)")
    print_stats(all_records,     "ALL COMBINED")

    print("\nDone. Metadata saved to:")
    print(f"    {PROCESSED_DIR}")
    print(f"    {SPLIT_DIR}")


if __name__ == "__main__":
    main()
