"""
Rhythm Diversity Score metric.

Spec:  D_rhythm = #unique_durations / #total_notes

Durations are quantised to the nearest 50 ms before counting to
avoid floating-point noise inflating the uniqueness count.

Usage:
    from src.evaluation.rhythm_score import rhythm_diversity_score
"""
from typing import List
import pretty_midi


def rhythm_diversity_score(notes: List[pretty_midi.Note]) -> float:
    """
    Compute rhythm diversity as the fraction of unique note durations.

    Each note duration (end - start, seconds) is rounded to the nearest
    50 ms (0.05 s) as specified.  Higher values indicate greater rhythmic
    variety; a score near 0 means most notes share the same duration.
    """
    if not notes:
        return 0.0
    # Quantise to nearest 50 ms
    durations = [round(round((n.end - n.start) / 0.05) * 0.05, 10) for n in notes]
    return len(set(durations)) / len(durations)
