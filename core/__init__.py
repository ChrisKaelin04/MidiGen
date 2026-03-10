"""Core data models and processing pipeline for MidiGen."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioData:
    """Raw audio data loaded from a file."""
    samples: np.ndarray       # mono float32 array
    sample_rate: int
    duration_sec: float
    file_path: Path


@dataclass
class NoteEvent:
    """A single detected note from transcription or processing."""
    pitch: int                # MIDI note number (0-127)
    start_sec: float
    end_sec: float
    amplitude: float          # 0.0-1.0 from basic-pitch confidence/amplitude


@dataclass
class ProcessingConfig:
    """Configuration for the audio-to-MIDI processing pipeline."""
    bpm: float
    start_sec: float
    end_sec: float
    note_low: int             # MIDI note number (e.g. 36 = C2)
    note_high: int            # MIDI note number (e.g. 84 = C6)
    confidence_threshold: float
    filter_ghost_notes: bool
    dynamic_velocity: bool
    preserve_durations: bool      # True = keep detected lengths, False = force 1/16th
    # basic-pitch model parameters
    onset_threshold: float        # model confidence for note onsets (lower = more sensitive)
    frame_threshold: float        # model confidence for sustained frames (lower = longer notes)
    minimum_note_length_ms: float # minimum note length in ms (lower = catches faster notes)
    # DSP enhancements
    ensemble_passes: int          # number of multi-pass runs (1 = single, 3-5 = ensemble)
    use_hpss: bool                # apply harmonic/percussive separation before transcription
    filter_harmonics: bool        # remove detected overtone/harmonic notes
    harmonic_filter_mode: str = "adaptive"  # "off", "on", "adaptive"
    # Fragment merging
    merge_fragments: bool = False         # stitch adjacent same-pitch fragments
    fragment_gap_tol: float = 0.05        # max gap (seconds) between fragments to merge
    fragment_reattack_ratio: float = 1.5  # amplitude ratio to detect re-attacks (0=disabled)
    # Quantization grid
    quantize_grid: str = "1/16"   # "1/8", "1/16", "1/32", "1/8T", "1/16T"
    # Pattern filling
    fill_patterns: bool = False           # fill missing notes in repeating patterns
    pattern_min_reps: int = 4             # minimum repetitions to detect a pattern
    pattern_fill_threshold: float = 0.75  # fraction of reps a pitch must appear in
    # Global timing offset in grid steps — positive shifts notes later
    # 1.0 = one grid unit (e.g. 1/16th note), computed from BPM + grid
    timing_offset_grid: float = 0.0
    # Melodia trick — basic-pitch post-processing to clean pitch contours
    melodia_trick: bool = True
    # Velocity curve exponent (1.0=linear, <1=boost quiet, >1=compress dynamics)
    velocity_curve: float = 1.0
