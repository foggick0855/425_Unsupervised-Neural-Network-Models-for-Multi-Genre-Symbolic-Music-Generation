"""
Evaluation metrics (spec Section 5):

  Pitch Histogram Similarity  H(p,q) = sum_{i=1}^{12} |p_i - q_i|
  Rhythm Diversity Score      D_rhythm = #unique_durations / #total_notes
  Repetition Ratio            R = #repeated_patterns / #total_patterns
  Human Listening Score       Score_human in [1, 5]

Usage:
  from src.evaluation.metrics import evaluate_midi, compare_models, print_report
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pretty_midi

try:
    from music21 import converter as m21_converter, analysis as m21_analysis
    _MUSIC21_AVAILABLE = True
except ImportError:
    _MUSIC21_AVAILABLE = False


# ---------------------------------------------------------------------------
# music21 — key estimation
# ---------------------------------------------------------------------------
def estimate_key(midi_path: str) -> Optional[str]:
    """
    Estimate the musical key of a MIDI file using music21's Krumhansl-Schmuckler
    key-finding algorithm.  Returns a string such as 'C major' or 'A minor',
    or None if music21 is unavailable or the file cannot be parsed.
    """
    if not _MUSIC21_AVAILABLE:
        return None
    try:
        score = m21_converter.parse(midi_path)
        key   = score.analyze("key")
        return str(key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-MIDI metric functions
# ---------------------------------------------------------------------------
def extract_notes(midi_path: str) -> List[pretty_midi.Note]:
    """Return all non-drum notes from a MIDI file."""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        return [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    except Exception:
        return []


# Pitch histogram and rhythm score logic live in their own modules
# (required by the project structure spec) and are imported here for
# backwards-compatible use throughout this file.
from src.evaluation.pitch_histogram import pitch_histogram, pitch_histogram_similarity, pitch_histogram_distance
from src.evaluation.rhythm_score    import rhythm_diversity_score


def repetition_ratio(notes: List[pretty_midi.Note], n: int = 4) -> float:
    """
    R = #repeated_n-grams / #total_n-grams  (pitch n-grams)
    """
    if len(notes) < n + 1:
        return 0.0
    pitches = [note.pitch for note in notes]
    ngrams  = [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]
    counts  = Counter(ngrams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / max(len(ngrams), 1)


def pitch_entropy(notes: List[pretty_midi.Note]) -> float:
    """Normalised Shannon entropy of 12-bin pitch histogram (0–1)."""
    hist = pitch_histogram(notes)
    ent  = -sum(p * math.log(p + 1e-9) for p in hist)
    return ent / math.log(12)


def evaluate_midi(midi_path: str, reference_path: Optional[str] = None) -> Dict:
    """
    Compute all metrics for one generated MIDI file.

    Args:
        midi_path      : path to the generated MIDI
        reference_path : optional path to a reference MIDI (for pitch similarity)
    """
    notes = extract_notes(midi_path)
    result = {
        "path":              midi_path,
        "n_notes":           len(notes),
        "rhythm_diversity":  rhythm_diversity_score(notes),
        "repetition_ratio":  repetition_ratio(notes),
        "pitch_entropy":     pitch_entropy(notes),
    }
    if reference_path:
        result["pitch_histogram_distance"] = pitch_histogram_distance(midi_path, reference_path)
    key = estimate_key(midi_path)
    if key is not None:
        result["estimated_key"] = key
    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
def evaluate_directory(midi_dir: str, reference_path: Optional[str] = None) -> List[Dict]:
    """Evaluate all MIDI files in a directory."""
    results = []
    midi_dir = Path(midi_dir)
    for path in sorted(midi_dir.glob("*.mid")):
        r = evaluate_midi(str(path), reference_path)
        results.append(r)
    return results


def aggregate(results: List[Dict]) -> Dict:
    """Average metrics across a list of evaluation results."""
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(model_dirs: Dict[str, str], reference_path: Optional[str] = None) -> Dict:
    """
    Compare multiple model output directories.

    Args:
        model_dirs : {"Model Name": "path/to/generated_midis/"}
        reference  : optional reference MIDI for pitch similarity
    Returns:
        dict of {model_name: averaged_metrics}
    """
    comparison = {}
    for name, directory in model_dirs.items():
        results = evaluate_directory(directory, reference_path)
        if results:
            comparison[name] = aggregate(results)
        else:
            comparison[name] = {"error": "no MIDI files found"}
    return comparison


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(comparison: Dict) -> None:
    """Pretty-print a comparison table to stdout."""
    if not comparison:
        print("No results to report.")
        return

    # Collect all metric keys
    all_keys = []
    for v in comparison.values():
        for k in v:
            if k not in all_keys and k != "error":
                all_keys.append(k)

    col_w = max(len(n) for n in comparison) + 2
    met_w = max((len(k) for k in all_keys), default=6) + 2

    header = f"{'Model':<{col_w}}" + "".join(f"{k:>{met_w}}" for k in all_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name, metrics in comparison.items():
        row = f"{name:<{col_w}}"
        for k in all_keys:
            val = metrics.get(k, float("nan"))
            row += f"{val:>{met_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")


def save_report(comparison: Dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Evaluation report saved -> {out_path}")


# ---------------------------------------------------------------------------
# Human score loader
# ---------------------------------------------------------------------------
def load_human_scores(scores_path: str) -> Dict:
    """
    Load human listening survey results.

    Expected JSON format:
    {
      "scores": [4.2, 3.8, 4.5, ...],   # one per generated sample
      "mean": 4.2,
      "participants": 10
    }
    """
    path = Path(scores_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    scores = data.get("scores", [])
    if scores:
        data["mean"] = sum(scores) / len(scores)
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.config import PROCESSED_DIR

    OUTPUTS = Path(__file__).parent.parent.parent / "outputs"

    model_dirs = {
        "AE (Task 1)":          str(OUTPUTS / "generated_midis" / "ae"),
        "VAE (Task 2)":         str(OUTPUTS / "generated_midis" / "vae"),
        "Transformer (Task 3)": str(OUTPUTS / "generated_midis" / "transformer"),
    }

    # Use first MAESTRO val file as reference
    try:
        ref = json.loads((PROCESSED_DIR / "maestro_metadata.json").read_text())
        reference = ref[0]["path"] if ref else None
    except Exception:
        reference = None

    comparison = compare_models(model_dirs, reference)
    print_report(comparison)
    save_report(comparison, str(OUTPUTS / "plots" / "evaluation_report.json"))
