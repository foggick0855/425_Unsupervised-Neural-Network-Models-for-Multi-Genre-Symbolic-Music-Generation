"""
Pitch histogram computation and similarity metric.

Spec:  H(p, q) = sum_{i=1}^{12} |p_i - q_i|

Usage:
    from src.evaluation.pitch_histogram import pitch_histogram, pitch_histogram_similarity
"""
from typing import List
import pretty_midi


def pitch_histogram(notes: List[pretty_midi.Note]) -> List[float]:
    """
    Normalised 12-bin pitch-class histogram.

    Maps each note to pitch % 12 (C=0 … B=11), counts occurrences,
    and normalises by total note count so the result sums to 1.
    """
    if not notes:
        return [0.0] * 12
    counts = [0] * 12
    for n in notes:
        counts[n.pitch % 12] += 1
    total = sum(counts)
    return [c / total for c in counts]


def pitch_histogram_distance(midi_path_1: str, midi_path_2: str) -> float:
    """
    Compute L1 distance between pitch-class histograms of two MIDI files.

    Spec formula:  H(p, q) = sum_{i=1}^{12} |p_i - q_i|
    Range: 0 (identical pitch usage) to 2 (completely disjoint).
    Lower is better.
    """
    def _extract(path: str) -> List[pretty_midi.Note]:
        try:
            midi = pretty_midi.PrettyMIDI(path)
            return [n for inst in midi.instruments if not inst.is_drum
                    for n in inst.notes]
        except Exception:
            return []

    h1 = pitch_histogram(_extract(midi_path_1))
    h2 = pitch_histogram(_extract(midi_path_2))
    return sum(abs(p - q) for p, q in zip(h1, h2))


def pitch_histogram_similarity(midi_path_1: str, midi_path_2: str) -> float:
    """Backwards-compatible alias — returns distance (lower = better)."""
    return pitch_histogram_distance(midi_path_1, midi_path_2)
