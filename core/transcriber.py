"""Wrap basic-pitch ML transcription, returning a list of NoteEvent.

Supports single-pass and multi-pass ensemble modes. Ensemble mode runs
the model multiple times with varying sensitivity and keeps notes that
appear consistently across passes (majority vote).
"""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core import AudioData, NoteEvent


# Ensemble presets: (onset_threshold, frame_threshold, minimum_note_length_ms)
# Ordered from conservative to aggressive
_ENSEMBLE_PRESETS = [
    (0.60, 0.40, 80),    # conservative — high confidence only
    (0.50, 0.30, 58),    # default
    (0.40, 0.25, 45),    # sensitive
    (0.30, 0.18, 35),    # aggressive
    (0.22, 0.12, 25),    # very aggressive
]


def _run_predict(
    audio_path: str,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
) -> list[NoteEvent]:
    """Run a single basic-pitch prediction pass."""
    from basic_pitch.inference import predict

    _, _, note_events = predict(
        audio_path=audio_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
    )

    return [
        NoteEvent(
            pitch=int(event[2]),
            start_sec=float(event[0]),
            end_sec=float(event[1]),
            amplitude=float(event[3]),
        )
        for event in note_events
    ]


def _notes_match(a: NoteEvent, b: NoteEvent, onset_tol: float = 0.05) -> bool:
    """Check if two notes match (same pitch, onset within tolerance)."""
    return a.pitch == b.pitch and abs(a.start_sec - b.start_sec) <= onset_tol


def _ensemble_merge(
    all_passes: list[list[NoteEvent]],
    min_votes: int,
    onset_tol: float = 0.05,
) -> list[NoteEvent]:
    """Merge notes from multiple passes using majority voting.

    A note is kept if it appears in at least min_votes passes.
    When kept, its timing and amplitude are averaged across the passes
    that detected it.

    Args:
        all_passes: List of note lists, one per pass.
        min_votes: Minimum number of passes a note must appear in to be kept.
        onset_tol: Onset tolerance in seconds for matching notes across passes.

    Returns:
        Merged list of NoteEvent with averaged properties.
    """
    if not all_passes:
        return []

    # Use the first pass as candidates, then check additional passes
    # Build a unified candidate pool from ALL passes
    candidates: list[dict] = []

    for pass_notes in all_passes:
        for note in pass_notes:
            # Check if this note already exists in candidates
            found = False
            for cand in candidates:
                if _notes_match(note, cand['ref'], onset_tol):
                    cand['votes'] += 1
                    cand['starts'].append(note.start_sec)
                    cand['ends'].append(note.end_sec)
                    cand['amps'].append(note.amplitude)
                    found = True
                    break
            if not found:
                candidates.append({
                    'ref': note,
                    'votes': 1,
                    'starts': [note.start_sec],
                    'ends': [note.end_sec],
                    'amps': [note.amplitude],
                })

    # Keep notes that meet the vote threshold
    result: list[NoteEvent] = []
    for cand in candidates:
        if cand['votes'] >= min_votes:
            result.append(NoteEvent(
                pitch=cand['ref'].pitch,
                start_sec=float(np.median(cand['starts'])),
                end_sec=float(np.median(cand['ends'])),
                amplitude=float(np.mean(cand['amps'])),
            ))

    return result


def transcribe(
    audio: AudioData,
    confidence_threshold: float = 0.5,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 58,
    ensemble_passes: int = 1,
) -> list[NoteEvent]:
    """Transcribe audio to a list of note events using basic-pitch.

    In single-pass mode (ensemble_passes=1), runs the model once with
    the specified parameters.

    In ensemble mode (ensemble_passes>1), runs the model multiple times
    with varying sensitivity presets and keeps notes that appear in a
    majority of passes. This reduces false positives (overtones, artifacts)
    while maintaining recall for real notes.

    Does NOT filter by note range — that is midi_processor's job.

    Args:
        audio: AudioData containing mono float32 samples.
        confidence_threshold: Minimum amplitude (0.0-1.0) to keep a note.
        onset_threshold: Model confidence for note onsets (single-pass only).
        frame_threshold: Model confidence for sustain frames (single-pass only).
        minimum_note_length_ms: Minimum note length in ms (single-pass only).
        ensemble_passes: Number of passes to run. 1=single-pass, 3-5=ensemble.

    Returns:
        List of NoteEvent with pitch, timing, and amplitude information.
    """
    # Write audio to temp file (basic-pitch requires a file path)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), audio.samples, audio.sample_rate)

        if ensemble_passes <= 1:
            # Single pass with user-specified parameters
            notes = _run_predict(
                str(tmp_path), onset_threshold, frame_threshold,
                minimum_note_length_ms,
            )
        else:
            # Multi-pass ensemble
            n = min(ensemble_passes, len(_ENSEMBLE_PRESETS))
            # Pick evenly spaced presets if fewer passes requested
            if n < len(_ENSEMBLE_PRESETS):
                indices = np.linspace(0, len(_ENSEMBLE_PRESETS) - 1, n, dtype=int)
                presets = [_ENSEMBLE_PRESETS[i] for i in indices]
            else:
                presets = list(_ENSEMBLE_PRESETS)

            all_passes: list[list[NoteEvent]] = []
            for onset, frame, min_len in presets:
                pass_notes = _run_predict(str(tmp_path), onset, frame, min_len)
                all_passes.append(pass_notes)

            # Require majority vote
            min_votes = max(2, (n // 2) + 1)
            notes = _ensemble_merge(all_passes, min_votes=min_votes)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Apply amplitude filter
    if confidence_threshold > 0:
        notes = [n for n in notes if n.amplitude >= confidence_threshold]

    return notes
